"""
``TrainResult.stop_reason`` and ``TrainResult.n_evaluations`` (issue #431).

``stop_reason`` names what ended the optimizer's loop, in snake_case;
``n_evaluations`` counts the candidates scored as a fitness, as
``TrainProgress.evaluations`` does: the final batch of ``final_shots`` is not
included. Every run here is seeded on the native simulator.
"""

import pytest

DIMENSIONS = 2
STOP_REASONS = {
    "fitness_tolerance",
    "parameter_tolerance",
    "iteration_budget",
    "evaluation_budget",
    "callback",
    "degenerate",
    "optimizer_finished",
    "walltime",
    "interrupted",
    "failed",
}
OPTIMIZERS = [
    "rand1bin",
    "shade",
    "lshade",
    "pso",
    "qng",
    "cmaes",
    "adam",
    "spsa",
    "rotosolve",
    "external",
]


def _method(name):
    """A small configuration of ``name`` whose budget ends the run."""
    import polypus
    from ask_tell_helpers import RandomSearch

    if name == "pso":
        return polypus.PSO(generations=4, population_size=6, tolerance=1e-12)
    if name == "qng":
        return polypus.QNG(lambda theta, a: 1.0, max_iters=4)
    if name == "cmaes":
        return polypus.CMAES(generations=4, tolerance=1e-12, patience=100)
    if name == "adam":
        return polypus.Adam(max_iters=4, patience=100)
    if name == "spsa":
        return polypus.SPSA(max_iters=4)
    if name == "rotosolve":
        return polypus.Rotosolve(sweeps=2, tolerance=0.0)
    if name == "external":
        return polypus.External(RandomSearch(DIMENSIONS, sizes=(4,), seed=3), 16)
    return polypus.DE(
        generations=4, population_size=6, strategy=name, tolerance=1e-12, patience=100
    )


# L-SHADE's schedule and External's max_evaluations are evaluation budgets.
BUDGET_REASON = {name: "iteration_budget" for name in OPTIMIZERS}
BUDGET_REASON.update(lshade="evaluation_budget", external="evaluation_budget")


def _train(method, **kwargs):
    import polypus

    P = polypus.Param
    circuit = polypus.Circuit(2).ry(0, P(0)).ry(1, P(1)).cx(0, 1).measure_all()
    return polypus.train(
        circuit,
        method,
        shots=64,
        n_qpus=1,
        dimensions=DIMENSIONS,
        expectation_function=lambda bits: float(bits.count("1")),
        infrastructure="local",
        nodes=1,
        cores_per_qpu=1,
        id="train_result",
        backend="polypus",
        seed=11,
        **kwargs,
    )


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_an_exhausted_budget_is_the_stop_reason(optimizer):
    result = _train(_method(optimizer))
    assert result.stop_reason in STOP_REASONS
    assert result.stop_reason == BUDGET_REASON[optimizer]
    assert isinstance(result.n_evaluations, int)
    assert result.n_evaluations > 0


def test_de_scores_its_initial_population_and_one_per_generation():
    import polypus

    population_size = 6
    result = _train(
        polypus.DE(
            generations=5,
            population_size=population_size,
            tolerance=1e-12,
            patience=100,
        )
    )
    # One batch for the initial population, then one per generation run.
    assert result.n_evaluations == population_size * (result.iterations_run + 1)


def test_a_fitness_stagnation_stop_is_reported():
    import polypus

    # No gain of 100 over one generation is possible: the second generation stops.
    result = _train(
        polypus.DE(generations=20, population_size=6, tolerance=100.0, patience=1)
    )
    assert result.stop_reason == "fitness_tolerance"
    assert result.converged
    assert result.iterations_run < 20


def test_a_callback_stop_is_reported_with_the_evaluations_it_saw():
    import polypus

    seen = []

    def stop_at_two(progress):
        seen.append(progress.evaluations)
        return progress.iteration == 2

    result = _train(
        polypus.DE(generations=5, population_size=6, tolerance=1e-12, patience=100),
        callback=stop_at_two,
    )
    assert result.stop_reason == "callback"
    assert result.iterations_run == 2
    assert result.n_evaluations == seen[-1]


def test_the_final_batch_is_not_counted():
    import polypus

    def method():
        return polypus.DE(
            generations=3, population_size=6, tolerance=1e-12, patience=100
        )

    plain = _train(method())
    final = _train(method(), final_shots=256)
    assert final.n_evaluations == plain.n_evaluations
    assert final.stop_reason == plain.stop_reason


def test_model_fit_reports_them_too():
    import polypus
    from polypus.qml import Ansatz, Encoder, Model, Readout, Z

    model = Model(
        Encoder(2).angle_encoder("ry"), Ansatz(2).hardware_efficient(1), Readout([Z(0)])
    )
    result = model.fit(
        [[0.1, 0.2], [0.4, 0.3]],
        method=polypus.DE(
            generations=3, population_size=6, tolerance=1e-12, patience=100
        ),
        seed=11,
    )
    assert result.stop_reason == "iteration_budget"
    assert result.n_evaluations == 6 * (result.iterations_run + 1)


def test_repr_shows_them():
    import polypus

    result = _train(
        polypus.DE(generations=3, population_size=6, tolerance=1e-12, patience=100)
    )
    text = repr(result)
    assert 'stop_reason="iteration_budget"' in text
    assert f"n_evaluations={result.n_evaluations}," in text


def _fit_minibatches(method, callback):
    from polypus.qml import Ansatz, Encoder, Model, Readout, Z

    model = Model(
        Encoder(2).angle_encoder("ry"), Ansatz(2).hardware_efficient(1), Readout([Z(0)])
    )
    return model.fit(
        [[0.1, 0.2], [0.4, 0.3], [2.9, 2.8], [0.2, 0.1]],
        method=method,
        batch_size=2,
        seed=11,
        callback=callback,
    )


def test_a_minibatch_rescore_is_counted():
    import polypus

    # DE scores minibatch estimates, so once it stops best_params is
    # re-scored on every sample: one candidate more than the callback saw.
    seen = []
    result = _fit_minibatches(
        polypus.DE(generations=3, population_size=6, tolerance=1e-12, patience=100),
        lambda progress: seen.append(progress.evaluations),
    )
    assert seen[-1] == 6 * (result.iterations_run + 1)
    assert result.n_evaluations == seen[-1] + 1


def test_exact_gradient_circuits_are_not_counted():
    import polypus

    # Adam measures θ with its parameter-shift circuits on every sample and
    # tracks its best there: θ counts, the shifted circuits and a re-score
    # do not.
    seen = []
    result = _fit_minibatches(
        polypus.Adam(max_iters=4, patience=100),
        lambda progress: seen.append(progress.evaluations),
    )
    assert seen == [1, 2, 3, 5]
    assert result.n_evaluations == 5
