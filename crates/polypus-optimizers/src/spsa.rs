//! Simultaneous perturbation stochastic approximation (SPSA), with a
//! calibrated first step.
//!
//! References:
//!
//! - J. C. Spall, "Multivariate stochastic approximation using a simultaneous
//!   perturbation gradient approximation", IEEE Transactions on Automatic
//!   Control 37(3), 332–341 (1992): the gradient estimate from two evaluations
//!   along a random perturbation, whatever the number of parameters, and the
//!   conditions on the gains that [`AlgorithmSPSAArgs`]'s preconditions keep.
//! - J. C. Spall, "Implementation of the simultaneous perturbation algorithm
//!   for stochastic optimization", IEEE Transactions on Aerospace and
//!   Electronic Systems 34(3), 817–823 (1998): the gains `a_k = a/(k + A)^α`
//!   and `c_k = c/k^γ`, the practical `α = 0.602` and `γ = 0.101`, ±1
//!   (Bernoulli) perturbations, and averaging several perturbations per
//!   iteration.
//! - B. D. M. Jones, L. Mineh and A. Montanaro, "Benchmarking a wide range of
//!   optimisers for solving the Fermi-Hubbard model using the variational
//!   quantum eigensolver", Quantum Sci. Technol. 10(4), 045032 (2025),
//!   arXiv:2411.13742: SPSA never evaluates its own iterate, so the iterate is
//!   measured every few iterations to know which one to return.
//!
//! The calibration follows Spall's guideline (1998): choose `a` from the
//! magnitude of the gradient estimate at θ₀, averaged over several
//! replications, so that the early steps have the size wanted. Here that size
//! is `target_step` and the replications are 25 perturbations sent in one
//! batch. It behaves as Qiskit's `SPSA.calibrate` does, without taking its
//! code. The fallback `a = target_step`, when no magnitude is usable, is
//! Polypus's own choice.

use crate::error::OptimizerError;
use crate::objective::EvaluationOracle;
use crate::outcome::{OptimizationOutcome, Optimizer};
use crate::rng::with_seeded_rng;
use crate::util::{check_initial_point, check_oracle_len, BestMeasured};
use rand::{Rng, RngExt};
use std::f64::consts::TAU;

/// Simultaneous perturbation stochastic approximation (Spall, 1992).
///
/// SPSA climbs the fitness along a gradient estimated from two evaluations
/// per random perturbation, whatever the number of parameters: with a ±1
/// perturbation `Δ` and a perturbation size `c_k`,
/// `ĝ = (f(θ + c_k·Δ) − f(θ − c_k·Δ)) / (2·c_k) · Δ`, each component of `Δ`
/// being its own inverse. It suits many parameters and few QPUs, where a
/// finite difference's `2·d` circuits per step would be slow.
///
/// # The run
///
/// 1. **θ₀** is [`initial_point`](AlgorithmSPSAArgs::initial_point) when
///    given, otherwise drawn uniformly from `[0, 2π)` (the parameters are
///    angles; SPSA takes no bounds). The iterates are not wrapped afterwards.
/// 2. **Calibration**: one batch of `2 × 25` candidates,
///    `[θ₀ + c·Δ₁, θ₀ − c·Δ₁, …, θ₀ + c·Δ₂₅, θ₀ − c·Δ₂₅]`, with `c` the
///    [`perturbation`](AlgorithmSPSAArgs::perturbation). Each pair gives the
///    magnitude `|f₊ − f₋| / (2c)`, which every component of that pair's
///    estimate `ĝᵢ` shares, `Δᵢ` being ±1; their mean is `mean|ĝ|`, and
///    `a = target_step / mean|ĝ|`. The first gain `a₁` is `a`, so the first
///    step moves each parameter by about
///    [`target_step`](AlgorithmSPSAArgs::target_step). A pair whose magnitude
///    is not finite (a non-finite value, or an overflow) is left out of the
///    mean. When no magnitude is left, their mean is zero (a flat landscape
///    around θ₀) or not finite, or `a` is not finite, `a = target_step`, as if
///    `mean|ĝ|` were 1, and a warning is logged; the run goes on.
/// 3. **Iterations** `k = 1, …, max_iters`, with Spall's gains
///    `a_k = a / k^α` and `c_k = c / k^γ` (`A = 0`): `K =`
///    [`resamplings`](AlgorithmSPSAArgs::resamplings) fresh ±1 perturbations,
///    one `evaluate_batch` of `2K` candidates
///    `[θ_k + c_k·Δ₁, θ_k − c_k·Δ₁, …]`, and the step
///    `θ_{k+1} = θ_k + a_k·ĝ_k` uphill, `ĝ_k` the mean of the `K` estimates
///    (those whose slope `(f₊ − f₋) / (2c_k)` is not finite left out; with
///    none left, θ stays).
///
/// # Measuring θ (contract C-5)
///
/// SPSA never evaluates θ_k for its step, so θ_k heads its iteration's batch
/// (`2K + 1` candidates, no extra call) on the first iteration and every
/// [`eval_interval`](AlgorithmSPSAArgs::eval_interval) iterations after it
/// (`k = 1, 1 + eval_interval, …`). After the last step, one batch of one
/// candidate measures the θ it reached, so the last iteration's pairs feed a
/// measured step. `best_params` is the measured θ with the best fitness and
/// `best_fitness` the value the oracle returned for it (a `NaN` never wins);
/// the perturbed candidates never count. `fitness_history` holds, for each
/// iteration, the best measured up to it, the final θ counted in the last
/// entry: the first iteration measures θ₀, so every entry is defined, and
/// the history ends at `best_fitness`.
///
/// There is no early stop: the run makes `max_iters` iterations and, like
/// [`AlgorithmQNG`](crate::AlgorithmQNG), reports `converged = false`.
///
/// # Reproducibility
///
/// The gains call `powf`, whose last bit Rust leaves to the platform's math
/// library, so a seeded run reproduces on the same platform and build, like
/// [`AlgorithmCmaEs`](crate::AlgorithmCmaEs), but may differ on another
/// (contract C-7).
///
/// # Example
///
/// ```
/// use polypus_optimizers::{AlgorithmSPSA, AlgorithmSPSAArgs, EvaluationOracle, Optimizer};
///
/// // Fitness Σ cos(xᵢ − 1) is maximised (value 4) at xᵢ = 1 (mod 2π).
/// struct Cosines;
/// impl EvaluationOracle for Cosines {
///     fn evaluate_batch(&self, candidates: &[Vec<f64>]) -> Vec<f64> {
///         candidates
///             .iter()
///             .map(|c| c.iter().map(|x| (x - 1.0).cos()).sum())
///             .collect()
///     }
/// }
///
/// let outcome = AlgorithmSPSA
///     .optimize(AlgorithmSPSAArgs {
///         oracle: Box::new(Cosines),
///         max_iters: 600,
///         resamplings: 1,
///         perturbation: 0.1,
///         target_step: 0.63,
///         alpha: AlgorithmSPSA::DEFAULT_ALPHA,
///         gamma: AlgorithmSPSA::DEFAULT_GAMMA,
///         eval_interval: 20,
///         dimensions: 4,
///         seed: Some(7),
///         initial_point: None,
///     })
///     .expect("valid arguments optimize successfully");
///
/// assert!(outcome.best_fitness > 4.0 - 1e-3, "fitness = {}", outcome.best_fitness);
/// assert_eq!(outcome.iterations_run, 600);
/// assert!(!outcome.converged);
/// ```
pub struct AlgorithmSPSA;

/// Arguments for [`AlgorithmSPSA`].
///
/// SPSA needs only an [`EvaluationOracle`]: no gradient and no variance
/// oracle.
///
/// # Preconditions
///
/// Checked before any RNG draw or oracle call, each violation an
/// [`OptimizerError::InvalidConfig`] naming the parameter: `max_iters >= 1`
/// (the first iteration measures θ₀, so zero iterations would measure
/// nothing), `dimensions >= 1`, `resamplings >= 1`, `perturbation` and
/// `target_step` finite and `> 0`, `alpha` in `(0.5, 1]`, `gamma > 0` with
/// `alpha − gamma > 0.5`, `eval_interval >= 1`, and `initial_point`, when
/// given, `dimensions` finite values. The ranges of `alpha` and `gamma` are
/// Spall's (1992) conditions on the gains: `c_k → 0` (`γ > 0`),
/// `Σ a_k = ∞` (`α ≤ 1`) and `Σ (a_k/c_k)² < ∞` (`α − γ > 1/2`).
pub struct AlgorithmSPSAArgs {
    /// Maps parameter vectors to fitness values (higher is better).
    pub oracle: Box<dyn EvaluationOracle>,
    /// Iterations, each one `evaluate_batch` call of `2·resamplings`
    /// candidates, plus θ_k when it is measured, and one step. The final θ is
    /// one call of one candidate more.
    pub max_iters: u32,
    /// Perturbations `K` per iteration: each iteration's batch holds `2K`
    /// candidates (`2K + 1` when it measures θ_k), and its gradient estimate
    /// is the mean of the `K` estimates. A `K` that fills the QPUs costs no
    /// more sequential rounds than one perturbation.
    pub resamplings: u32,
    /// The perturbation size `c`, in radians: the calibration's, and
    /// `c_k = c / k^γ` in iteration `k`.
    pub perturbation: f64,
    /// The magnitude, in radians, of each parameter's first step, which sets
    /// `a` through the calibration.
    pub target_step: f64,
    /// Decay exponent `α` of the step gain `a_k = a / k^α`.
    pub alpha: f64,
    /// Decay exponent `γ` of the perturbation `c_k = c / k^γ`.
    pub gamma: f64,
    /// θ_k is measured, first in its iteration's batch, on the first
    /// iteration and every `eval_interval` iterations after it; the final θ
    /// is measured whatever the interval.
    pub eval_interval: u32,
    /// Number of parameters.
    pub dimensions: u32,
    /// Optional RNG seed. `None` (the default) uses [`rand::rng`];
    /// `Some(seed)` makes the run reproducible.
    pub seed: Option<u64>,
    /// Optional θ₀, one value per dimension, all finite, taken as it is: the
    /// calibration batch is centred on it and it is the first candidate of the
    /// first iteration. θ₀ is drawn from `[0, 2π)` either way and then
    /// replaced, so the rest of the run draws what it would without it.
    /// `None` starts from the drawn θ₀.
    pub initial_point: Option<Vec<f64>>,
}

impl AlgorithmSPSA {
    /// Perturbations in the calibration batch, which holds twice as many
    /// candidates.
    pub const CALIBRATION_PERTURBATIONS: u32 = 25;
    /// The default `alpha`: Spall's (1998) practical value.
    pub const DEFAULT_ALPHA: f64 = 0.602;
    /// The default `gamma`: Spall's (1998) practical value.
    pub const DEFAULT_GAMMA: f64 = 0.101;

    /// The optimizer's human-readable name.
    pub fn name(&self) -> String {
        String::from("SPSA")
    }

    /// A short description of the optimizer.
    pub fn description(&self) -> String {
        String::from(
            "Trains a variational quantum circuit with simultaneous perturbation stochastic \
             approximation (SPSA): gradient ascent along an estimate from two evaluations per \
             random perturbation, whatever the number of parameters, with Spall's decaying \
             gains and a first step calibrated at the starting point.",
        )
    }

    /// Check the preconditions of [`AlgorithmSPSAArgs`] that
    /// [`optimize`](Optimizer::optimize) checks before any RNG draw or oracle
    /// call, in the same order, without running anything: the configuration
    /// listed on [`AlgorithmSPSAArgs`], then `initial_point` (`dimensions`
    /// finite values). `train` and `qml.train` call it before any backend
    /// exists.
    pub fn check_preconditions(
        max_iters: u32,
        resamplings: u32,
        perturbation: f64,
        target_step: f64,
        alpha: f64,
        gamma: f64,
        eval_interval: u32,
        dimensions: u32,
        initial_point: Option<&[f64]>,
    ) -> Result<(), OptimizerError> {
        validate(
            max_iters,
            dimensions,
            resamplings,
            perturbation,
            target_step,
            alpha,
            gamma,
            eval_interval,
        )?;
        check_initial_point(initial_point, dimensions as usize, None)
    }

    /// Whether iteration `k` (from 1) measures its θ: the first, and every
    /// `eval_interval` after it.
    fn measures(k: u32, eval_interval: u32) -> bool {
        (k - 1) % eval_interval == 0
    }

    fn run_with_rng<R: Rng>(
        args: AlgorithmSPSAArgs,
        rng: &mut R,
    ) -> Result<OptimizationOutcome, OptimizerError> {
        let AlgorithmSPSAArgs {
            oracle,
            max_iters,
            resamplings,
            perturbation,
            target_step,
            alpha,
            gamma,
            eval_interval,
            dimensions,
            seed: _,
            initial_point,
        } = args;

        Self::check_preconditions(
            max_iters,
            resamplings,
            perturbation,
            target_step,
            alpha,
            gamma,
            eval_interval,
            dimensions,
            initial_point.as_deref(),
        )?;
        let dims = dimensions as usize;
        // Draw θ₀ in [0, 2π), or start at the caller's point: θ₀ is drawn
        // either way, so the RNG stream does not depend on it.
        let drawn: Vec<f64> = (0..dims).map(|_| rng.random_range(0.0..TAU)).collect();
        let mut theta = initial_point.unwrap_or(drawn);

        // ── Calibration: one batch of 25 pairs around θ₀ sets `a` ───────────────
        let deltas: Vec<Vec<f64>> = (0..Self::CALIBRATION_PERTURBATIONS)
            .map(|_| rademacher(rng, dims))
            .collect();
        let batch = perturbed(&theta, perturbation, &deltas);
        let values = oracle.evaluate_batch(&batch);
        check_oracle_len(batch.len(), values.len())?;
        let a = calibrated_gain(&values, perturbation, target_step);
        log::debug!("Calibration: a = {a:.6e}");

        let resamplings = resamplings as usize;
        let mut best = BestMeasured::new();
        // Best measured fitness after each iteration (see
        // `OptimizationOutcome::fitness_history`).
        let mut fitness_history: Vec<f64> = Vec::with_capacity(max_iters as usize);

        for k in 1..=max_iters {
            let a_k = a / f64::from(k).powf(alpha);
            let c_k = perturbation / f64::from(k).powf(gamma);
            let deltas: Vec<Vec<f64>> = (0..resamplings).map(|_| rademacher(rng, dims)).collect();
            let measured = Self::measures(k, eval_interval);

            // ── 1. One batch: θ_k when it is measured, then the K pairs ─────────
            let mut batch = Vec::with_capacity(2 * resamplings + usize::from(measured));
            if measured {
                batch.push(theta.clone());
            }
            batch.extend(perturbed(&theta, c_k, &deltas));
            let values = oracle.evaluate_batch(&batch);
            check_oracle_len(batch.len(), values.len())?;
            let pairs = if measured {
                best.offer(&theta, values[0]);
                log::debug!("Iteration {k}: fitness {:.4}", values[0]);
                &values[1..]
            } else {
                &values[..]
            };
            // The running best of the measured θs: the first iteration measures
            // θ₀, so no entry is the `-∞` of nothing measured (C-5).
            fitness_history.push(best.value());

            // ── 2. Step uphill along the mean of the K estimates ─────────────────
            let mut gradient = vec![0.0_f64; dims];
            let mut used = 0usize;
            for (delta, pair) in deltas.iter().zip(pairs.chunks_exact(2)) {
                let slope = (pair[0] - pair[1]) / (2.0 * c_k);
                if !slope.is_finite() {
                    continue;
                }
                used += 1;
                // Each component of Δ is ±1, its own inverse.
                for (g, d) in gradient.iter_mut().zip(delta) {
                    *g += slope * d;
                }
            }
            if used > 0 {
                let count = used as f64;
                for (x, g) in theta.iter_mut().zip(&gradient) {
                    *x += a_k * (g / count);
                }
            }
        }

        // ── Measure the θ the last step reached ─────────────────────────────────
        // One candidate, so the last pairs do not go into an unmeasured step;
        // its value joins the last iteration's entry, which ends the history at
        // the reported best (C-5).
        let values = oracle.evaluate_batch(std::slice::from_ref(&theta));
        check_oracle_len(1, values.len())?;
        log::debug!("Final θ: fitness {:.4}", values[0]);
        best.offer(&theta, values[0]);
        if let Some(last) = fitness_history.last_mut() {
            *last = best.value();
        }

        // `max_iters >= 1` and the first iteration measures θ₀, so `best` holds
        // a measured θ.
        let (best_params, best_fitness) = best.into_parts();
        Ok(OptimizationOutcome {
            best_params,
            best_fitness,
            iterations_run: max_iters as usize,
            converged: false,
            fitness_history,
        })
    }
}

/// A perturbation of `dims` independent ±1 components, each with probability
/// 1/2 (Spall's symmetric Bernoulli distribution).
fn rademacher<R: Rng>(rng: &mut R, dims: usize) -> Vec<f64> {
    (0..dims)
        .map(|_| if rng.random_bool(0.5) { 1.0 } else { -1.0 })
        .collect()
}

/// `[θ + c·Δ₁, θ − c·Δ₁, θ + c·Δ₂, …]`.
fn perturbed(theta: &[f64], c: f64, deltas: &[Vec<f64>]) -> Vec<Vec<f64>> {
    deltas
        .iter()
        .flat_map(|delta| {
            let plus = theta.iter().zip(delta).map(|(x, d)| x + c * d).collect();
            let minus = theta.iter().zip(delta).map(|(x, d)| x - c * d).collect();
            [plus, minus]
        })
        .collect()
}

/// The step gain `a` from the calibration batch's `values`, pairs of
/// `f(θ₀ + c·Δᵢ)` and `f(θ₀ − c·Δᵢ)`: `target_step / mean|ĝ|`, the mean over
/// the pairs whose magnitude `|f₊ − f₋| / (2c)` is finite. Without a usable
/// mean (a flat landscape, or no finite value), `target_step`.
fn calibrated_gain(values: &[f64], c: f64, target_step: f64) -> f64 {
    let magnitudes: Vec<f64> = values
        .chunks_exact(2)
        .map(|pair| ((pair[0] - pair[1]) / (2.0 * c)).abs())
        .filter(|m| m.is_finite())
        .collect();
    if !magnitudes.is_empty() {
        let mean = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;
        let a = target_step / mean;
        if mean.is_finite() && mean > 0.0 && a.is_finite() {
            return a;
        }
    }
    log::warn!(
        "SPSA calibration: the {} gradient estimates at the starting point have no usable \
         magnitude ({} of them finite); using a = target_step = {target_step}",
        values.len() / 2,
        magnitudes.len(),
    );
    target_step
}

/// Reject a configuration SPSA cannot run, before any RNG draw or oracle call.
fn validate(
    max_iters: u32,
    dimensions: u32,
    resamplings: u32,
    perturbation: f64,
    target_step: f64,
    alpha: f64,
    gamma: f64,
    eval_interval: u32,
) -> Result<(), OptimizerError> {
    let invalid = |parameter: &'static str, reason: String| {
        Err(OptimizerError::InvalidConfig { parameter, reason })
    };
    if max_iters == 0 {
        return invalid(
            "max_iters",
            "must be at least 1 (0 iterations never measures the starting point)".to_string(),
        );
    }
    if dimensions == 0 {
        return invalid(
            "dimensions",
            "must be at least 1 (there is nothing to optimise)".to_string(),
        );
    }
    if resamplings == 0 {
        return invalid(
            "resamplings",
            "must be at least 1 (each iteration needs a perturbation)".to_string(),
        );
    }
    for (parameter, value) in [("perturbation", perturbation), ("target_step", target_step)] {
        if !value.is_finite() || value <= 0.0 {
            return invalid(parameter, format!("must be finite and > 0, got {value}"));
        }
    }
    if !(alpha > 0.5 && alpha <= 1.0) {
        return invalid(
            "alpha",
            format!(
                "must be in (0.5, 1], Spall's conditions on the step gain a/k^alpha \
                 (its sum diverges, and alpha - gamma > 0.5), got {alpha}"
            ),
        );
    }
    if !(gamma > 0.0 && alpha - gamma > 0.5) {
        return invalid(
            "gamma",
            format!(
                "must be > 0 with alpha - gamma > 0.5 (alpha = {alpha}), Spall's conditions \
                 that the perturbation c/k^gamma shrinks and that (a_k/c_k)^2 has a finite \
                 sum, got {gamma}"
            ),
        );
    }
    if eval_interval == 0 {
        return invalid(
            "eval_interval",
            "must be at least 1 (iterations between measurements of the parameters)".to_string(),
        );
    }
    Ok(())
}

impl Optimizer for AlgorithmSPSA {
    type Args = AlgorithmSPSAArgs;

    fn optimize(&self, args: Self::Args) -> Result<OptimizationOutcome, OptimizerError> {
        with_seeded_rng(args.seed, |rng| Self::run_with_rng(args, rng))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theta_is_measured_first_and_every_interval() {
        let measured: Vec<u32> = (1..=23)
            .filter(|&k| AlgorithmSPSA::measures(k, 5))
            .collect();
        assert_eq!(measured, [1, 6, 11, 16, 21]);
        assert!((1..=4).all(|k| AlgorithmSPSA::measures(k, 1)));
        assert!(AlgorithmSPSA::measures(1, 20));
    }

    #[test]
    fn the_gain_is_the_target_over_the_mean_magnitude() {
        // Magnitudes |f₊ − f₋| / (2c) with c = 0.5: 1, 3 and 2.
        let values = [2.0, 1.0, -1.0, 2.0, 0.0, 2.0];
        assert_eq!(calibrated_gain(&values, 0.5, 0.6), 0.6 / 2.0);
    }

    #[test]
    fn a_pair_with_a_non_finite_value_is_left_out_of_the_calibration() {
        let values = [
            2.0,
            1.0,
            f64::NAN,
            2.0,
            0.0,
            3.0,
            f64::INFINITY,
            f64::INFINITY,
        ];
        // Magnitudes 1 and 3 (c = 0.5); the NaN and ∞ − ∞ pairs are left out.
        assert_eq!(calibrated_gain(&values, 0.5, 0.6), 0.6 / 2.0);
    }

    #[test]
    fn a_flat_or_unusable_calibration_falls_back_to_the_target_step() {
        assert_eq!(calibrated_gain(&[1.0; 50], 0.1, 0.63), 0.63);
        assert_eq!(calibrated_gain(&[f64::NAN; 50], 0.1, 0.63), 0.63);
        // A magnitude so small that a would overflow.
        assert_eq!(calibrated_gain(&[1e-320, 0.0], 0.5, 0.63), 0.63);
    }
}
