#!/usr/bin/env python
"""Report stage: turn the accumulated results CSV into metrics + figures + report.

Fully independent of the run/sweep stages — it only reads the CSV, so a report
can be regenerated any number of times without re-running any QAOA. It combines
every row it finds (one or several CSVs from repeated sweeps), validates the
schema loudly (a corrupt row aborts the report rather than being dropped), then
writes:

* aggregated metrics per (n_qubits, method): mean / std / median / min / max and
  a 95% confidence interval for both approximation ratio and run time, with the
  effective repetition count ``n``;
* four figures (dataviz reference palette, consistent across all of them);
* ``maxcut_report.md`` (Markdown tables + referenced PNGs, for repo/PR review) and
  ``maxcut_report.html`` (self-contained: CSS + figures embedded inline).

Example::

    python examples/max_cut/report_maxcut.py                 # default CSV + out dir
    python examples/max_cut/report_maxcut.py --csv a.csv b.csv --out-dir /tmp/rep
"""

import argparse
import base64
import logging
import math
import os
import re
import sys
from collections import OrderedDict, defaultdict

import numpy as np

import maxcut_lib as mc

log = logging.getLogger("maxcut.report")


# ─────────────────────────────────────────────────────────────────────────────
# Method presentation — colours are the dataviz reference categorical palette
# (slots 1/2/3/6, validated: worst adjacent CVD ΔE 37.7). Colour follows the
# method (the entity), never its rank; scipy is additionally dashed/hatched as
# a secondary encoding so the classical baseline reads apart from Polypus.
# ─────────────────────────────────────────────────────────────────────────────

METHOD_STYLE = OrderedDict([
    ("polypus_local", ("Polypus DE", "#2a78d6", "-", "o", "")),
    ("polypus_local_pso", ("Polypus PSO", "#1baf7a", "-", "s", "")),
    ("polypus_local_qng", ("Polypus QNG", "#eda100", "-", "^", "")),
    ("scipy", ("scipy DE (baseline)", "#e34948", "--", "D", "///")),
    # CUNQA variants (manual only) reuse the optimizer colour with a distinct marker.
    ("polypus_cunqa", ("Polypus DE (CUNQA)", "#2a78d6", ":", "o", "")),
    ("polypus_cunqa_pso", ("Polypus PSO (CUNQA)", "#1baf7a", ":", "s", "")),
    ("polypus_cunqa_qng", ("Polypus QNG (CUNQA)", "#eda100", ":", "^", "")),
])
_FALLBACK_STYLE = ("?", "#898781", "-", "x", "")

# Light-surface chrome / ink from the reference palette.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _style(method):
    return METHOD_STYLE.get(method, _FALLBACK_STYLE)


def _method_order(methods):
    """Present known methods in the palette order, then any unknown ones."""
    known = [m for m in METHOD_STYLE if m in methods]
    unknown = sorted(m for m in methods if m not in METHOD_STYLE)
    return known + unknown


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────


def _stats(values, lo=None, hi=None):
    """Summary statistics plus a 95% t-interval for the mean.

    ``mean``/``std`` are the raw sample values (never altered). The *displayed*
    interval (``ci_low``/``ci_high`` and the asymmetric error offsets
    ``err_low``/``err_high``) is clipped to the metric's physical domain
    ``[lo, hi]`` — an unclipped t-interval can otherwise report a ratio above 1
    or a negative time with few repetitions, which reads as a bug in a report
    meant to be shown to others. Pass ``lo``/``hi`` (either may be ``None`` for
    an open side); the clip only bounds what is drawn, not the statistics.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean())
    if n > 1:
        std = float(arr.std(ddof=1))
        try:
            from scipy.stats import t

            half = float(t.ppf(0.975, n - 1) * std / math.sqrt(n))
        except Exception:
            half = float(1.96 * std / math.sqrt(n))
    else:
        std = 0.0
        half = 0.0
    ci_low, ci_high = mean - half, mean + half
    if lo is not None:
        ci_low = max(lo, ci_low)
    if hi is not None:
        ci_high = min(hi, ci_high)
    return {
        "n": int(n),
        "mean": mean,
        "std": std,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "ci_half": half,           # raw half-width (unclipped), for reference
        "ci_low": ci_low,          # displayed bound, clipped to [lo, hi]
        "ci_high": ci_high,
        "err_low": mean - ci_low,  # asymmetric error-bar offsets (>= 0)
        "err_high": ci_high - mean,
    }


def aggregate(rows):
    """Group rows by (method, n_qubits) and compute ratio/time statistics.

    Returns an ordered list of dicts, sorted by method (palette order) then
    n_qubits.
    """
    groups = defaultdict(lambda: {"ratio": [], "time": []})
    for r in rows:
        key = (r["method"], r["n_qubits"])
        groups[key]["ratio"].append(r["approximation_ratio"])
        groups[key]["time"].append(r["time_seconds"])

    methods = _method_order({m for m, _ in groups})
    out = []
    for method in methods:
        for q in sorted({q for m, q in groups if m == method}):
            g = groups[(method, q)]
            out.append({
                "method": method,
                "display": _style(method)[0],
                "n_qubits": q,
                # Clip displayed CIs to each metric's physical domain: ratio is
                # a fraction in [0, 1]; run time is non-negative.
                "ratio": _stats(g["ratio"], lo=0.0, hi=1.0),
                "time": _stats(g["time"], lo=0.0, hi=None),
            })
    return out


def _provenance(rows):
    def distinct(field):
        return sorted({r[field] for r in rows})

    return {
        "total_runs": len(rows),
        "methods": _method_order({r["method"] for r in rows}),
        "qubits": distinct("n_qubits"),
        "base_seeds": distinct("base_seed"),
        "seeds": distinct("seed"),
        "repeats": len(distinct("repeat_index")),
        "git_commits": distinct("git_commit"),
        "polypus_versions": distinct("polypus_version"),
        "shots": distinct("n_shots"),
        "timestamps": (min(distinct("timestamp")), max(distinct("timestamp"))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training configuration (schema v2) — the hyper-parameters behind the metrics.
#
# Same "fail high, never hide" stance as read_records: a value that should be
# constant but is not (a scaling factor differing between rows, a per-method
# knob that drifted) is surfaced with a ⚠ marker and a logged warning — never
# silently collapsed to one representative value.
# ─────────────────────────────────────────────────────────────────────────────

#: (label, field) for the sweep-wide scaling factors + shot count.
_CONFIG_FACTORS = [
    ("layers_factor", "layers_factor"),
    ("population_factor", "population_factor"),
    ("generations_factor", "generations_factor"),
    ("n_shots", "n_shots"),
]

#: Per-n_qubits problem sizing (derived from the factors + n_qubits).
_SIZING_FIELDS = ["layers", "dimensions", "population_size", "max_generations",
                  "best_solution_bruteforce"]
_SIZING_HEADERS = ["qubits", "layers", "dimensions", "population_size",
                   "max_generations", "optimal cut (bruteforce)"]


def _fmt_val(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _distinct_vals(rows, field):
    """Distinct values of ``field`` across ``rows``, deduped, None sorted last."""
    seen = []
    for r in rows:
        v = r[field]
        if v not in seen:
            seen.append(v)
    return sorted(seen, key=lambda v: (v is None, v))


def _cell(vals):
    """(text, varies) for a set of distinct values. ``varies`` is True when more
    than one distinct value was seen where one was expected."""
    if len(vals) <= 1:
        return (_fmt_val(vals[0]) if vals else "n/a"), False
    return "varies: " + ", ".join(_fmt_val(v) for v in vals), True


def _method_param_fields(method):
    """(label, field) hyper-parameters that the given method actually uses."""
    opt = mc.METHOD_SPECS[method][0]
    if opt == "qng":
        return [("learning_rate", "qng_learning_rate"),
                ("finite-diff step", "qng_step"),
                ("tikhonov reg", "qng_tikhonov")]
    if opt == "scipy":
        # scipy's differential_evolution takes popsize=population_factor (NOT
        # population_size); see the note under the sizing table.
        return [("tolerance", "tolerance"),
                ("popsize (→ differential_evolution)", "population_factor"),
                ("scipy_workers", "scipy_workers")]
    return [("tolerance", "tolerance")]  # de / pso


def _config(rows):
    """Structured training configuration for the report's Configuration section."""
    factors = []
    for label, field in _CONFIG_FACTORS:
        text, varies = _cell(_distinct_vals(rows, field))
        factors.append((label, text, varies))

    per_method = []
    for method in _method_order({r["method"] for r in rows}):
        mrows = [r for r in rows if r["method"] == method]
        params = []
        for label, field in _method_param_fields(method):
            text, varies = _cell(_distinct_vals(mrows, field))
            params.append((label, text, varies))
        per_method.append({"method": method, "display": _style(method)[0], "params": params})

    sizing = []
    for q in sorted({r["n_qubits"] for r in rows}):
        qrows = [r for r in rows if r["n_qubits"] == q]
        cells = {f: _cell(_distinct_vals(qrows, f)) for f in _SIZING_FIELDS}
        sizing.append({"n_qubits": q, "cells": cells})

    return {"factors": factors, "per_method": per_method, "sizing": sizing}


def _config_warnings(config):
    """Human-readable messages for every value flagged as unexpectedly varying."""
    msgs = []
    for label, _text, varies in config["factors"]:
        if varies:
            msgs.append(f"scaling factor/shots '{label}' is not constant across the CSV")
    for pm in config["per_method"]:
        for label, _text, varies in pm["params"]:
            if varies:
                msgs.append(f"hyper-parameter '{label}' varies within method '{pm['display']}'")
    for s in config["sizing"]:
        for field, (_text, varies) in s["cells"].items():
            if varies:
                msgs.append(f"problem sizing '{field}' varies at {s['n_qubits']} qubits")
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# Figures (matplotlib, Agg, light reference surface)
# ─────────────────────────────────────────────────────────────────────────────


def _init_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": INK,
        "grid.color": GRID,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def _series_by_method(agg, field):
    """method -> (qubits[], mean[], err_low[], err_high[]) for the given metric.

    The errors are the asymmetric, domain-clipped CI offsets, so an error bar
    never crosses the metric's physical bound (e.g. a ratio bar stops at 1.0).
    """
    series = OrderedDict()
    for method in _method_order({row["method"] for row in agg}):
        pts = sorted((r["n_qubits"], r[field]) for r in agg if r["method"] == method)
        series[method] = (
            [q for q, _ in pts],
            [s["mean"] for _, s in pts],
            [s["err_low"] for _, s in pts],
            [s["err_high"] for _, s in pts],
        )
    return series


def _fig_metric_vs_qubits(plt, agg, field, ylabel, title, path, logy=False):
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=140)
    series = _series_by_method(agg, field)
    for method, (xs, ys, err_lo, err_hi) in series.items():
        name, color, ls, marker, _ = _style(method)
        ax.errorbar(xs, ys, yerr=[err_lo, err_hi], label=name, color=color, linestyle=ls,
                    marker=marker, markersize=7, linewidth=2, capsize=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
    # Identity comes from the legend: the DE and scipy lines run nearly on top of
    # each other, so per-line end labels would collide — the legend is cleaner.
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("qubits (graph nodes)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, loc="left")
    all_q = sorted({q for r in agg for q in [r["n_qubits"]]})
    ax.set_xticks(all_q)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_ratio_distribution(plt, rows, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=140)
    methods = _method_order({r["method"] for r in rows})
    data = [[r["approximation_ratio"] for r in rows if r["method"] == m] for m in methods]
    labels = [_style(m)[0] for m in methods]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops=dict(color=INK, linewidth=1.5),
                    flierprops=dict(marker="o", markersize=4, markerfacecolor=MUTED,
                                    markeredgecolor=SURFACE))
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for patch, method in zip(bp["boxes"], methods):
        _, color, _, _, hatch = _style(method)
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor(SURFACE)
        patch.set_linewidth(1.5)
        if hatch:
            patch.set_hatch(hatch)
    ax.set_ylabel("approximation ratio")
    ax.set_title("Approximation-ratio distribution per method (all qubit sizes pooled)",
                 fontsize=12, loc="left")
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_polypus_vs_scipy(plt, agg, path):
    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=140)
    methods = _method_order({r["method"] for r in agg})
    qubits = sorted({r["n_qubits"] for r in agg})
    ratio_by = {(r["method"], r["n_qubits"]): r["ratio"] for r in agg}
    x = np.arange(len(qubits))
    n_m = max(1, len(methods))
    width = 0.8 / n_m
    for i, method in enumerate(methods):
        name, color, _, _, hatch = _style(method)
        means = [ratio_by.get((method, q), {}).get("mean", np.nan) for q in qubits]
        err_lo = [ratio_by.get((method, q), {}).get("err_low", 0.0) for q in qubits]
        err_hi = [ratio_by.get((method, q), {}).get("err_high", 0.0) for q in qubits]
        offs = x + (i - (n_m - 1) / 2) * width
        ax.bar(offs, means, width=width * 0.9, label=name, color=color,
               edgecolor=SURFACE, linewidth=1.2, hatch=hatch or None,
               yerr=[err_lo, err_hi], ecolor=MUTED, capsize=2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in qubits])
    ax.set_xlabel("qubits (graph nodes)")
    ax.set_ylabel("mean approximation ratio")
    ax.set_ylim(0, 1.05)
    ax.set_title("Polypus optimizers vs. scipy baseline (mean ratio, 95% CI)",
                 fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_figures(agg, rows, out_dir):
    plt = _init_matplotlib()
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    times = [r["time"]["mean"] for r in agg if r["time"]["mean"] > 0]
    logy = bool(times) and (max(times) / min(times) > 20)

    figures = OrderedDict()
    figures["ratio_vs_qubits"] = os.path.join(fig_dir, "ratio_vs_qubits.png")
    _fig_metric_vs_qubits(plt, agg, "ratio", "approximation ratio (mean, 95% CI)",
                          "Approximation ratio vs. qubits", figures["ratio_vs_qubits"])

    figures["time_vs_qubits"] = os.path.join(fig_dir, "time_vs_qubits.png")
    _fig_metric_vs_qubits(plt, agg, "time",
                          "run time [s]" + (" (log)" if logy else ""),
                          "Run time vs. qubits", figures["time_vs_qubits"], logy=logy)

    figures["ratio_distribution"] = os.path.join(fig_dir, "ratio_distribution.png")
    _fig_ratio_distribution(plt, rows, figures["ratio_distribution"])

    figures["polypus_vs_scipy"] = os.path.join(fig_dir, "polypus_vs_scipy.png")
    _fig_polypus_vs_scipy(plt, agg, figures["polypus_vs_scipy"])
    return figures


# ─────────────────────────────────────────────────────────────────────────────
# Metrics table + narrative
# ─────────────────────────────────────────────────────────────────────────────

_TABLE_HEADERS = [
    "method", "qubits", "n",
    "ratio mean", "ratio std", "ratio median", "ratio min", "ratio max", "ratio 95% CI",
    "time mean [s]", "time std [s]", "time 95% CI [s]",
]


def _table_rows(agg):
    for r in agg:
        rt, tm = r["ratio"], r["time"]
        yield [
            r["display"], str(r["n_qubits"]), str(rt["n"]),
            f"{rt['mean']:.4f}", f"{rt['std']:.4f}", f"{rt['median']:.4f}",
            f"{rt['min']:.4f}", f"{rt['max']:.4f}",
            f"[{rt['ci_low']:.4f}, {rt['ci_high']:.4f}]",
            f"{tm['mean']:.3f}", f"{tm['std']:.3f}",
            f"[{tm['ci_low']:.3f}, {tm['ci_high']:.3f}]",
        ]


def _headline(agg, prov):
    """A short factual comparison of the best Polypus method vs. scipy at the
    largest qubit count, if both are present. Returns None otherwise."""
    if not agg:
        return None
    q = max(prov["qubits"])
    at_q = {r["method"]: r for r in agg if r["n_qubits"] == q}
    if "scipy" not in at_q:
        return None
    polypus = [m for m in at_q if m.startswith("polypus")]
    if not polypus:
        return None
    best = max(polypus, key=lambda m: at_q[m]["ratio"]["mean"])
    scipy = at_q["scipy"]
    best_r = at_q[best]["ratio"]["mean"]
    scipy_r = scipy["ratio"]["mean"]
    speedup = scipy["time"]["mean"] / at_q[best]["time"]["mean"] if at_q[best]["time"]["mean"] else float("nan")
    return {
        "q": q, "best": _style(best)[0],
        "best_ratio": best_r, "scipy_ratio": scipy_r,
        "ratio_delta": best_r - scipy_r,
        "best_time": at_q[best]["time"]["mean"], "scipy_time": scipy["time"]["mean"],
        "speedup": speedup,
    }


def _md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _flag(text, varies):
    """Append the ⚠ marker to a cell whose value varied unexpectedly."""
    return text + (" ⚠" if varies else "")


_CONFIG_INTRO = (
    "The training configuration behind the metrics above. The scaling factors "
    "and shot count are expected to be constant across the whole sweep; the "
    "problem sizing is derived from them and `n_qubits`. A **⚠** marks a value "
    "that is not constant where one was expected — it is shown as a list of the "
    "distinct values rather than silently collapsed to one (same fail-high "
    "stance as the CSV reader)."
)

_POPSIZE_NOTE = (
    "`population_size` is the population handed to the **Polypus** DE/PSO "
    "optimizers (`dimensions × population_factor`). The **scipy** baseline "
    "instead receives `popsize = population_factor` and multiplies it by "
    "`dimensions` internally, so its per-method row reports `popsize` while the "
    "sizing table reports `population_size` — two conventions for (effectively) "
    "the same total population, not the same number."
)


def _config_md(config):
    out = ["## Configuration", "", _CONFIG_INTRO, "",
           "**Scaling factors & shots**", "",
           _md_table(["parameter", "value"],
                     [[label, _flag(text, varies)] for label, text, varies in config["factors"]]),
           "",
           "**Hyper-parameters per method** — only the knobs each optimizer "
           "actually uses; `n/a` means the method does not consult that knob.", ""]
    for pm in config["per_method"]:
        out += [f"*{pm['display']}*", "",
                _md_table(["hyper-parameter", "value"],
                          [[label, _flag(text, varies)] for label, text, varies in pm["params"]]),
                ""]
    out += ["**Problem sizing by qubit count** — varies with `n_qubits`; sets "
            "the scale of each run and gives context for the ratio/time rows above.", ""]
    sizing_rows = [
        [str(s["n_qubits"])] + [_flag(*s["cells"][f]) for f in _SIZING_FIELDS]
        for s in config["sizing"]
    ]
    out += [_md_table(_SIZING_HEADERS, sizing_rows), "",
            f"> {_POPSIZE_NOTE}"]
    return "\n".join(out)


def _narrative_md(prov, headline):
    seeds = prov["base_seeds"]
    seed_str = ", ".join(str(s) for s in seeds)
    parts = [
        "## What this experiment demonstrates",
        "",
        "MaxCut is optimised with QAOA on a local statevector/Aer simulator, comparing "
        "three Polypus optimizers (**DE**, **PSO**, **QNG**) against a pure-scipy "
        "`differential_evolution` baseline (\"no Polypus\"). Each cell below is the "
        "aggregate of several independent repetitions, so the numbers carry a spread, "
        "not a single lucky/unlucky run.",
        "",
        "## Reproducibility",
        "",
        f"The whole sweep is reproducible from a single base seed. Repetition *r* of "
        f"every (qubits, method) uses `seed = base_seed + r`, and every source of shot "
        f"randomness (training oracle, QNG variance callback, final evaluation, scipy's "
        f"Aer sampling) is seeded from it — so two runs with the same seed produce the "
        f"identical approximation ratio and best bitstring. This report's data used base "
        f"seed(s) **{seed_str}**. Regenerate the data with:",
        "",
        "```bash",
        f"python examples/max_cut/sweep_maxcut.py --qubits {' '.join(str(q) for q in prov['qubits'])} "
        f"--repeats {prov['repeats']} --seed {seeds[0]}",
        "python examples/max_cut/report_maxcut.py",
        "```",
        "",
        "## How to read the metrics table",
        "",
        "One row per (method, qubit count). `ratio mean` is the primary quality metric "
        "(expected cut ÷ optimal cut, so 1.0 is perfect); the 95% CI is a t-interval over "
        "the `n` repetitions. `time` columns are wall-clock seconds per run. Higher ratio "
        "is better; lower time is better.",
        "",
        "> The CI shown is a 95% t-interval **clipped to each metric's physical range** "
        "(ratio to `[0, 1]`, time to `[0, ∞)`). With few repetitions the raw t-interval can "
        "extend past those bounds; the mean and standard deviation are the unmodified sample "
        "values — only the displayed interval (and its error bars) is clamped.",
    ]
    if headline:
        h = headline
        cmp_word = "above" if h["ratio_delta"] >= 0 else "below"
        parts += [
            "",
            "## Headline",
            "",
            f"At {h['q']} qubits, the best Polypus optimizer (**{h['best']}**) reached a "
            f"mean approximation ratio of **{h['best_ratio']:.4f}**, {abs(h['ratio_delta']):.4f} "
            f"{cmp_word} the scipy baseline ({h['scipy_ratio']:.4f}), "
            f"with a mean run time of {h['best_time']:.2f}s vs scipy's {h['scipy_time']:.2f}s "
            f"(≈{h['speedup']:.2f}× time ratio).",
        ]
    return "\n".join(parts)


def render_markdown(agg, rows, figures, prov, config, out_path):
    headline = _headline(agg, prov)
    fig_titles = [
        ("ratio_vs_qubits", "Approximation ratio vs. qubits"),
        ("time_vs_qubits", "Run time vs. qubits"),
        ("ratio_distribution", "Approximation-ratio distribution per method"),
        ("polypus_vs_scipy", "Polypus vs. scipy baseline"),
    ]
    out = [
        "# MaxCut–QAOA experiment report",
        "",
        _narrative_md(prov, headline),
        "",
        "## Provenance",
        "",
        f"- runs: **{prov['total_runs']}** | qubits: {prov['qubits']} | "
        f"methods: {[_style(m)[0] for m in prov['methods']]}",
        f"- repeats: {prov['repeats']} | base seed(s): {prov['base_seeds']} | shots: {prov['shots']}",
        f"- git commit(s): {prov['git_commits']} | polypus: {prov['polypus_versions']}",
        f"- timestamps: {prov['timestamps'][0]} … {prov['timestamps'][1]}",
        "",
        _config_md(config),
        "",
        "## Metrics",
        "",
        _md_table(_TABLE_HEADERS, list(_table_rows(agg))),
        "",
        "## Figures",
        "",
    ]
    for key, title in fig_titles:
        rel = os.path.join("figures", os.path.basename(figures[key]))
        out += [f"### {title}", "", f"![{title}]({rel})", ""]
    with open(out_path, "w") as fh:
        fh.write("\n".join(out) + "\n")


def _b64_img(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _inline_md_to_html(text):
    """Convert the small subset of inline Markdown used in the shared config
    blurbs (``**bold**`` and `` `code` ``) to HTML, so the same source string
    renders correctly in both the ``.md`` and ``.html`` reports."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def _html_table(headers, rows):
    thead = "".join(f"<th>{h}</th>" for h in headers)
    tbody = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return (f'<div class="table-wrap"><table><thead><tr>{thead}</tr></thead>'
            f'<tbody>{tbody}</tbody></table></div>')


def _config_html(config):
    factor_tbl = _html_table(
        ["parameter", "value"],
        [[label, _flag(text, varies)] for label, text, varies in config["factors"]],
    )
    method_blocks = "".join(
        f'<h3 class="cfg-h3">{pm["display"]}</h3>'
        + _html_table(["hyper-parameter", "value"],
                      [[label, _flag(text, varies)] for label, text, varies in pm["params"]])
        for pm in config["per_method"]
    )
    sizing_tbl = _html_table(
        _SIZING_HEADERS,
        [[str(s["n_qubits"])] + [_flag(*s["cells"][f]) for f in _SIZING_FIELDS]
         for s in config["sizing"]],
    )
    return f"""<h2>Configuration</h2>
<p>{_inline_md_to_html(_CONFIG_INTRO)}</p>
<h3 class="cfg-h3">Scaling factors &amp; shots</h3>
{factor_tbl}
<h3 class="cfg-h3">Hyper-parameters per method</h3>
<p>Only the knobs each optimizer actually uses; <code>n/a</code> means the method does not consult that knob.</p>
{method_blocks}
<h3 class="cfg-h3">Problem sizing by qubit count</h3>
<p>These vary with <code>n_qubits</code> and set the scale of each run.</p>
{sizing_tbl}
<p class="prov">{_inline_md_to_html(_POPSIZE_NOTE)}</p>"""


def render_html(agg, rows, figures, prov, config, out_path):
    headline = _headline(agg, prov)
    thead = "".join(f"<th>{h}</th>" for h in _TABLE_HEADERS)
    tbody = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for row in _table_rows(agg)
    )
    fig_titles = [
        ("ratio_vs_qubits", "Approximation ratio vs. qubits"),
        ("time_vs_qubits", "Run time vs. qubits"),
        ("ratio_distribution", "Approximation-ratio distribution per method"),
        ("polypus_vs_scipy", "Polypus vs. scipy baseline"),
    ]
    figs_html = "".join(
        f'<figure><figcaption>{title}</figcaption>'
        f'<img alt="{title}" src="data:image/png;base64,{_b64_img(figures[key])}"></figure>'
        for key, title in fig_titles
    )
    headline_html = ""
    if headline:
        h = headline
        cmp_word = "above" if h["ratio_delta"] >= 0 else "below"
        headline_html = (
            f'<p class="headline">At <b>{h["q"]} qubits</b>, the best Polypus optimizer '
            f'(<b>{h["best"]}</b>) reached a mean approximation ratio of '
            f'<b>{h["best_ratio"]:.4f}</b>, {abs(h["ratio_delta"]):.4f} {cmp_word} the scipy '
            f'baseline ({h["scipy_ratio"]:.4f}); mean run time {h["best_time"]:.2f}s vs '
            f'{h["scipy_time"]:.2f}s (≈{h["speedup"]:.2f}× time ratio).</p>'
        )
    seed0 = prov["base_seeds"][0]
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MaxCut–QAOA experiment report</title>
<style>
  :root {{
    --surface: {SURFACE}; --plane: #f9f9f7; --ink: {INK}; --ink2: {INK2};
    --muted: {MUTED}; --grid: {GRID}; --axis: {AXIS};
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--plane); color: var(--ink);
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height: 1.55; }}
  main {{ max-width: 960px; margin: 0 auto; padding: 32px 20px 64px; }}
  h1 {{ font-size: 1.7rem; margin: 0 0 .2em; }}
  h2 {{ font-size: 1.2rem; margin: 1.8em 0 .5em; border-bottom: 1px solid var(--grid); padding-bottom: .2em; }}
  h3.cfg-h3 {{ font-size: .98rem; margin: 1.2em 0 .4em; color: var(--ink); }}
  p, li {{ color: var(--ink2); }}
  code, pre {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; }}
  pre {{ background: var(--surface); border: 1px solid var(--grid); border-radius: 8px;
         padding: 12px 14px; overflow-x: auto; font-size: .85rem; }}
  .headline {{ background: var(--surface); border-left: 4px solid #2a78d6; border-radius: 6px;
               padding: 12px 16px; color: var(--ink); }}
  .prov {{ font-size: .82rem; color: var(--muted); }}
  .table-wrap {{ overflow-x: auto; border: 1px solid var(--grid); border-radius: 8px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .82rem;
           font-variant-numeric: tabular-nums; background: var(--surface); }}
  th, td {{ padding: 6px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--grid); }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ position: sticky; top: 0; background: var(--surface); color: var(--ink);
              border-bottom: 2px solid var(--axis); }}
  tbody tr:hover {{ background: #f3f3f0; }}
  .figs {{ display: grid; grid-template-columns: 1fr; gap: 22px; margin-top: 8px; }}
  figure {{ margin: 0; background: var(--surface); border: 1px solid var(--grid); border-radius: 10px; padding: 10px 10px 4px; }}
  figcaption {{ font-size: .85rem; font-weight: 600; color: var(--ink); margin: 2px 4px 8px; }}
  img {{ display: block; width: 100%; height: auto; }}
  @media (min-width: 720px) {{ .figs {{ grid-template-columns: 1fr 1fr; }} }}
</style></head>
<body><main>
<h1>MaxCut–QAOA experiment report</h1>
{headline_html}
<h2>What this experiment demonstrates</h2>
<p>MaxCut is optimised with QAOA on a local statevector/Aer simulator, comparing three
Polypus optimizers (<b>DE</b>, <b>PSO</b>, <b>QNG</b>) against a pure-scipy
<code>differential_evolution</code> baseline (&ldquo;no&nbsp;Polypus&rdquo;). Each cell in the
metrics table aggregates several independent repetitions.</p>
<h2>Reproducibility</h2>
<p>The whole sweep is reproducible from a single base seed: repetition <i>r</i> of every
(qubits, method) uses <code>seed = base_seed + r</code>, and every source of shot randomness is
seeded from it, so the same seed reproduces the identical approximation ratio and best
bitstring. This report used base seed(s) <b>{', '.join(str(s) for s in prov['base_seeds'])}</b>.</p>
<pre>python examples/max_cut/sweep_maxcut.py --qubits {' '.join(str(q) for q in prov['qubits'])} --repeats {prov['repeats']} --seed {seed0}
python examples/max_cut/report_maxcut.py</pre>
<h2>Metrics</h2>
<p>One row per (method, qubit count). <code>ratio mean</code> is expected cut ÷ optimal cut
(1.0 is perfect); the 95% CI is a t-interval over the <code>n</code> repetitions. Higher ratio is
better; lower time is better.</p>
<p class="prov">The CI shown is a 95% t-interval <b>clipped to each metric's physical range</b>
(ratio to [0,&nbsp;1], time to [0,&nbsp;∞)). With few repetitions the raw t-interval can extend
past those bounds; the mean and standard deviation are the unmodified sample values — only the
displayed interval (and its error bars) is clamped.</p>
<div class="table-wrap"><table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table></div>
<h2>Provenance</h2>
<p class="prov">runs: {prov['total_runs']} · qubits: {prov['qubits']} · repeats: {prov['repeats']}
· base seed(s): {prov['base_seeds']} · shots: {prov['shots']} · git: {prov['git_commits']}
· polypus: {prov['polypus_versions']} · {prov['timestamps'][0]} … {prov['timestamps'][1]}</p>
{_config_html(config)}
<h2>Figures</h2>
<div class="figs">{figs_html}</div>
</main></body></html>
"""
    with open(out_path, "w") as fh:
        fh.write(html)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser():
    p = argparse.ArgumentParser(
        description="Generate the MaxCut-QAOA metrics + figures + report from the CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", nargs="+", default=[mc.DEFAULT_CSV],
                   help="One or more results CSVs to combine.")
    p.add_argument("--out-dir", default=mc.DEFAULT_REPORT_DIR, help="Output directory for the report.")
    p.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        rows = mc.read_records(args.csv)
    except mc.ExperimentError as exc:
        log.error("cannot read results (%s): %s", type(exc).__name__, exc)
        return 1

    agg = aggregate(rows)
    prov = _provenance(rows)
    config = _config(rows)
    for msg in _config_warnings(config):
        log.warning("configuration inconsistency (shown with ⚠ in the report): %s", msg)
    os.makedirs(args.out_dir, exist_ok=True)

    figures = render_figures(agg, rows, args.out_dir)
    md_path = os.path.join(args.out_dir, "maxcut_report.md")
    html_path = os.path.join(args.out_dir, "maxcut_report.html")
    render_markdown(agg, rows, figures, prov, config, md_path)
    render_html(agg, rows, figures, prov, config, html_path)

    log.info("read %d rows from %s", len(rows), args.csv)
    log.info("wrote figures: %s", ", ".join(os.path.basename(p) for p in figures.values()))
    log.info("wrote report: %s", md_path)
    log.info("wrote report: %s", html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
