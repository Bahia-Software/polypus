"""Typed execution targets (issue #434).

``polypus.Aer``, ``Native``, ``Cunqa``, ``Qmio`` and ``Registered`` describe
where a circuit runs; ``target=`` passes one to ``run_quantum_circuit``,
``train``, ``qml.train`` and ``qml.predict``. The ``infrastructure``/``backend``
strings stay as shorthand and resolve through one alias table:
``polypus._target_of`` exposes that resolution so every string combination
accepted today can be checked against exactly one typed target, and every
rejected one against the error the entry points raise.
"""

import sys

import pytest

pytest.importorskip("qiskit_aer")

import polypus  # noqa: E402
from fake_cunqa import FakeCunqa  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402

NOT_COMPILED = (
    "the 'qmio' infrastructure requires compiling polypus with --features qmio"
)


def _qmio_compiled():
    try:
        polypus._target_of("qmio")
    except polypus.ConfigurationError as err:
        if str(err) == NOT_COMPILED:
            return False
        raise
    return True


QMIO_COMPILED = _qmio_compiled()
NOISE = object()
COMMAND = {"command": [sys.executable, "worker.py"]}

# Every combination of strings and provider arguments the entry points accept,
# with the one typed target it maps to.
ACCEPTED = [
    # local Aer
    ({"infrastructure": "local"}, polypus.Aer()),
    ({"infrastructure": "local", "backend": "aer"}, polypus.Aer()),
    ({"infrastructure": "local", "backend": "AerSimulator"}, polypus.Aer()),
    (
        {"infrastructure": "local", "sim_method": "matrix_product_state"},
        polypus.Aer(sim_method="matrix_product_state"),
    ),
    (
        {"infrastructure": "local", "backend": "AerSimulator", "noise_model": NOISE},
        polypus.Aer(noise_model=NOISE),
    ),
    ({"infrastructure": "local", "fusion": False}, polypus.Aer()),
    # local ignores the CUNQA allocation, unchecked
    (
        {"infrastructure": "local", "nodes": 0, "cores_per_qpu": 0, "walltime": "x"},
        polypus.Aer(),
    ),
    # local native
    *[
        (
            {"infrastructure": "local", "backend": alias, "fusion": fusion},
            polypus.Native(fusion=fusion),
        )
        for alias in ("polypus", "statevector", "polypus_statevector")
        for fusion in (None, False, True)
    ],
    # cunqa
    ({"infrastructure": "cunqa"}, polypus.Cunqa()),
    ({"infrastructure": "cunqa", "backend": "AerSimulator"}, polypus.Cunqa()),
    (
        {
            "infrastructure": "cunqa",
            "nodes": 2,
            "cores_per_qpu": 4,
            "walltime": "48:00:00",
            "sim_method": "statevector",
            "fusion": False,
        },
        polypus.Cunqa(
            nodes=2, cores_per_qpu=4, walltime="48:00:00", sim_method="statevector"
        ),
    ),
    # registry
    ({"infrastructure": "subprocess"}, polypus.Registered("subprocess")),
    (
        {"infrastructure": "subprocess", "fusion": False},
        polypus.Registered("subprocess"),
    ),
    (
        {"infrastructure": "subprocess", "options": COMMAND},
        polypus.Registered("subprocess", options=COMMAND),
    ),
    (
        {
            "infrastructure": "subprocess",
            "options": {"command": tuple(COMMAND["command"])},
        },
        polypus.Registered("subprocess", options=COMMAND),
    ),
]

QMIO_ACCEPTED = [
    (
        {"infrastructure": "qmio", **({"backend": alias} if alias else {})},
        polypus.Qmio(program_format=fmt),
    )
    for alias, fmt in [
        (None, "openqasm"),
        ("aer", "openqasm"),
        ("openqasm", "openqasm"),
        ("qasm", "openqasm"),
        ("qmio", "openqasm"),
        ("qir", "qir"),
        ("qir_text", "qir"),
        ("qir_bitcode", "qir_bitcode"),
        ("qir_compiled", "qir_bitcode"),
    ]
]

# Combinations rejected today; `_target_of` must raise the same error as the
# entry points.
REJECTED = [
    {"infrastructure": "local", "backend": "does-not-exist"},
    {"infrastructure": "local", "backend": "polypus", "noise_model": NOISE},
    {"infrastructure": "local", "backend": "statevector", "sim_method": "statevector"},
    {"infrastructure": "local", "backend": "aer", "fusion": True},
    {"infrastructure": "local", "options": {"command": "x"}},
    {"infrastructure": "local", "options": {"k": 1}},
    {"infrastructure": "cunqa", "backend": "polypus"},
    {"infrastructure": "cunqa", "backend": "foo"},
    {"infrastructure": "cunqa", "noise_model": NOISE},
    {"infrastructure": "cunqa", "nodes": 0},
    {"infrastructure": "cunqa", "cores_per_qpu": 0},
    {"infrastructure": "cunqa", "walltime": "1-00:00:00"},
    {"infrastructure": "cunqa", "fusion": True},
    {"infrastructure": "cunqa", "options": {"command": "x"}},
    {"infrastructure": "subprocess", "noise_model": NOISE},
    {"infrastructure": "subprocess", "sim_method": "statevector"},
    {"infrastructure": "subprocess", "backend": "polypus"},
    {"infrastructure": "subprocess", "fusion": True},
    {"infrastructure": "nowhere"},
    {"infrastructure": "qmio", "backend": "bogus"},
    {"infrastructure": "qmio", "noise_model": NOISE},
    {"infrastructure": "qmio", "sim_method": "statevector"},
    {"infrastructure": "qmio", "fusion": True},
]


def _circuit():
    return polypus.Circuit(2).h(0).cx(0, 1).measure_all()


def _error_of(call):
    with pytest.raises(Exception) as info:
        call()
    return type(info.value), str(info.value)


def _qml_circuits():
    feature_map = QuantumCircuit(1)
    feature_map.ry(Parameter("x"), 0)
    ansatz = QuantumCircuit(1)
    ansatz.ry(Parameter("w"), 0)
    return feature_map, ansatz


def _train(**kwargs):
    kwargs.setdefault("method", polypus.DE(generations=2, population_size=4))
    return polypus.train(
        polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all(),
        shots=64,
        n_qpus=1,
        dimensions=1,
        expectation_function=lambda bits: float(bits == "1"),
        id="targets",
        seed=11,
        **kwargs,
    )


def _qml_train(**kwargs):
    feature_map, ansatz = _qml_circuits()
    return polypus.qml.train(
        feature_map,
        ansatz,
        [[0.1], [0.2]],
        polypus.DE(generations=1, population_size=4),
        shots=16,
        n_qpus=1,
        dimensions=1,
        expectation_function=lambda bits: float(bits == "1"),
        id="targets",
        seed=3,
        **kwargs,
    )


def _qml_predict(**kwargs):
    feature_map, ansatz = _qml_circuits()
    return polypus.qml.predict(
        feature_map, ansatz, [[0.1], [0.4]], [0.3], shots=64, **kwargs
    )


def _run(**kwargs):
    return polypus.run_quantum_circuit(_circuit(), shots=128, **kwargs)


class TestAliasTable:
    @pytest.mark.parametrize(("kwargs", "expected"), ACCEPTED, ids=repr)
    def test_every_accepted_combination_maps_to_one_target(self, kwargs, expected):
        assert polypus._target_of(**kwargs) == expected

    @pytest.mark.parametrize(("kwargs", "expected"), QMIO_ACCEPTED, ids=repr)
    def test_every_qmio_program_format_maps_to_one_target(self, kwargs, expected):
        """Accepted only in a build with `--features qmio`."""
        if QMIO_COMPILED:
            assert polypus._target_of(**kwargs) == expected
        else:
            with pytest.raises(polypus.ConfigurationError, match="--features qmio"):
                polypus._target_of(**kwargs)

    @pytest.mark.parametrize("kwargs", REJECTED, ids=repr)
    def test_a_rejected_combination_raises_what_the_entry_point_raises(self, kwargs):
        expected = _error_of(lambda: _run(**kwargs))
        assert _error_of(lambda: polypus._target_of(**kwargs)) == expected
        assert issubclass(expected[0], (ValueError, TypeError))


class TestConstruction:
    def test_defaults(self):
        assert polypus.Aer().sim_method == "automatic"
        assert polypus.Aer().noise_model is None
        assert polypus.Native().fusion is None
        cunqa = polypus.Cunqa()
        assert (cunqa.nodes, cunqa.cores_per_qpu, cunqa.walltime, cunqa.sim_method) == (
            1,
            2,
            None,
            "automatic",
        )
        assert polypus.Qmio().program_format == "openqasm"
        registered = polypus.Registered("subprocess")
        assert (registered.name, registered.options) == ("subprocess", {})

    def test_qmio_normalizes_the_program_format(self):
        assert polypus.Qmio(program_format="qir_text") == polypus.Qmio(
            program_format="qir"
        )
        assert polypus.Qmio(program_format="qasm").program_format == "openqasm"

    @pytest.mark.parametrize(
        ("build", "match"),
        [
            (lambda: polypus.Cunqa(nodes=0), "nodes must be >= 1"),
            (lambda: polypus.Cunqa(cores_per_qpu=0), "cores_per_qpu must be >= 1"),
            (lambda: polypus.Cunqa(walltime="1-00:00:00"), "walltime"),
            (
                lambda: polypus.Qmio(program_format="bogus"),
                "unknown qmio program format 'bogus'",
            ),
            (
                lambda: polypus.Qmio(program_format="aer"),
                "unknown qmio program format 'aer'",
            ),
            (lambda: polypus.Registered(""), "name must not be empty"),
            (lambda: polypus.Registered("local"), "use polypus.Aer"),
            (lambda: polypus.Registered("qmio"), "use polypus.Qmio"),
            (lambda: polypus.Registered("nowhere"), "unknown infrastructure 'nowhere'"),
        ],
    )
    def test_invalid_fields_are_configuration_errors(self, build, match):
        with pytest.raises(polypus.ConfigurationError, match=match):
            build()

    @pytest.mark.parametrize(
        "build",
        [
            lambda: polypus.Native(fusion="yes"),
            lambda: polypus.Aer(sim_method=1),
            lambda: polypus.Registered("subprocess", options={"k": 1}),
        ],
    )
    def test_wrong_types_are_type_errors(self, build):
        with pytest.raises(TypeError):
            build()

    def test_equality_and_repr(self):
        assert polypus.Native(fusion=False) != polypus.Native()
        assert polypus.Aer() != polypus.Native()
        assert repr(polypus.Cunqa(nodes=2)) == (
            "Cunqa(nodes=2, cores_per_qpu=2, walltime=None, sim_method='automatic')"
        )
        assert repr(polypus.Qmio()) == "Qmio(program_format='openqasm')"

    def test_fields_are_read_only(self):
        with pytest.raises(AttributeError):
            polypus.Native().fusion = True


class TestRunQuantumCircuit:
    @pytest.mark.parametrize(
        ("target", "strings"),
        [
            (polypus.Native(), {"infrastructure": "local", "backend": "polypus"}),
            (
                polypus.Native(fusion=False),
                {"infrastructure": "local", "backend": "polypus", "fusion": False},
            ),
            (polypus.Aer(), {"infrastructure": "local"}),
            (
                polypus.Aer(sim_method="statevector"),
                {"infrastructure": "local", "sim_method": "statevector"},
            ),
        ],
        ids=repr,
    )
    def test_a_target_reproduces_the_string_call(self, target, strings):
        typed = _run(target=target, seed=7)
        plain = _run(seed=7, **strings)
        assert typed.counts == plain.counts
        assert typed.merged_counts == plain.merged_counts
        assert (typed.backend, typed.infrastructure, typed.seed) == (
            "polypus" if isinstance(target, polypus.Native) else "aer",
            "local",
            7,
        )
        assert (plain.backend, plain.infrastructure) == (
            strings.get("backend", "aer"),
            "local",
        )

    def test_a_cunqa_target_reaches_qraise_like_the_strings(self, monkeypatch):
        def qraise_args(**kwargs):
            fake = FakeCunqa().install(monkeypatch)
            _run(seed=5, **kwargs)
            name, args = fake.calls[0]
            assert name == "qraise"
            return {k: v for k, v in args.items() if "fam" not in k}

        typed = qraise_args(
            target=polypus.Cunqa(nodes=2, cores_per_qpu=3, walltime="01:00:00")
        )
        plain = qraise_args(
            infrastructure="cunqa", nodes=2, cores_per_qpu=3, walltime="01:00:00"
        )
        assert typed == plain

    def test_a_qiskit_circuit_is_rejected_by_a_native_target(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()
        with pytest.raises(
            polypus.ConfigurationError, match="cannot execute a Qiskit QuantumCircuit"
        ):
            polypus.run_quantum_circuit(qc, shots=10, target=polypus.Native())

    def test_options_reach_a_registered_target(self):
        with pytest.raises(polypus.ConfigurationError, match="options"):
            _run(
                target=polypus.Registered("subprocess", options=COMMAND),
                options=COMMAND,
            )

    def test_options_are_rejected_on_a_typed_builtin(self):
        with pytest.raises(
            polypus.ConfigurationError, match="does not accept 'options'"
        ):
            _run(target=polypus.Aer(), options=COMMAND)

    def test_a_qmio_target_fails_like_the_strings(self, monkeypatch):
        """Without the feature, both fail alike; with it, both fail to reach a
        QPU without an endpoint."""
        monkeypatch.delenv("ZMQ_SERVER", raising=False)
        plain = _error_of(lambda: _run(infrastructure="qmio", backend="qir"))
        assert (
            _error_of(lambda: _run(target=polypus.Qmio(program_format="qir"))) == plain
        )

    def test_neither_target_nor_infrastructure_is_a_type_error(self):
        with pytest.raises(TypeError, match="'infrastructure'"):
            polypus.run_quantum_circuit(_circuit(), shots=10)

    def test_a_target_must_be_a_target_class(self):
        with pytest.raises(TypeError, match="target must be polypus.Aer"):
            _run(target="aer")


class TestTrain:
    def test_a_target_reproduces_the_string_call(self):
        typed = _train(target=polypus.Native())
        plain = _train(
            infrastructure="local", nodes=1, cores_per_qpu=2, backend="polypus"
        )
        assert typed.best_params == plain.best_params
        assert typed.best_fitness == plain.best_fitness
        assert typed.fitness_history == plain.fitness_history

    def test_a_positional_call_still_works(self):
        result = polypus.train(
            polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all(),
            polypus.DE(generations=1, population_size=4),
            16,
            1,
            1,
            lambda bits: float(bits == "1"),
            "local",
            1,
            2,
            "positional",
            seed=1,
        )
        assert len(result.best_params) == 1

    @pytest.mark.parametrize("missing", ["infrastructure", "nodes", "cores_per_qpu"])
    def test_a_string_call_still_needs_each_former_positional(self, missing):
        kwargs = {"infrastructure": "local", "nodes": 1, "cores_per_qpu": 2}
        del kwargs[missing]
        with pytest.raises(TypeError, match=f"'{missing}'"):
            _train(**kwargs)

    def test_id_is_still_required(self):
        with pytest.raises(TypeError, match="'id'"):
            polypus.train(
                polypus.Circuit(1).ry(0, polypus.Param(0)).measure_all(),
                polypus.DE(generations=1, population_size=4),
                shots=16,
                n_qpus=1,
                dimensions=1,
                expectation_function=lambda bits: float(bits == "1"),
                target=polypus.Native(),
            )


class TestQml:
    def test_predict_with_a_target_reproduces_the_string_call(self):
        typed = _qml_predict(target=polypus.Aer(), seed=4)
        plain = _qml_predict(infrastructure="local", seed=4)
        assert typed.counts == plain.counts
        assert (typed.backend, typed.infrastructure, typed.seed) == ("aer", "local", 4)

    def test_train_with_a_target_reproduces_the_string_call(self):
        typed = _qml_train(target=polypus.Aer())
        plain = _qml_train(infrastructure="local", nodes=1, cores_per_qpu=2)
        assert typed.best_params == plain.best_params
        assert typed.best_fitness == plain.best_fitness

    @pytest.mark.parametrize(
        "call", [_qml_train, _qml_predict], ids=["train", "predict"]
    )
    def test_a_native_target_is_rejected_like_the_string(self, call):
        with pytest.raises(
            polypus.ConfigurationError,
            match="native 'polypus' backend is not supported",
        ):
            call(target=polypus.Native())


# Each exclusive argument, with a non-default value, next to `target=`.
EXCLUSIVE = {
    "infrastructure": "local",
    "backend": "polypus",
    "sim_method": "statevector",
    "noise_model": NOISE,
    "fusion": False,
    "nodes": 2,
    "cores_per_qpu": 3,
    "walltime": "01:00:00",
}
ENTRY_POINTS = {
    "run_quantum_circuit": (_run, set(EXCLUSIVE)),
    "train": (_train, set(EXCLUSIVE)),
    "qml.train": (_qml_train, set(EXCLUSIVE) - {"fusion"}),
    "qml.predict": (_qml_predict, set(EXCLUSIVE) - {"fusion"}),
}


class TestExclusivity:
    @pytest.mark.parametrize(
        ("entry_point", "name"),
        [
            (entry, name)
            for entry, (_, names) in ENTRY_POINTS.items()
            for name in sorted(names)
        ],
    )
    def test_a_string_argument_next_to_a_target_is_rejected(
        self, monkeypatch, entry_point, name
    ):
        fake = FakeCunqa().install(monkeypatch)
        call, _ = ENTRY_POINTS[entry_point]
        with pytest.raises(
            polypus.ConfigurationError, match=f"target= cannot be combined with {name}"
        ):
            call(target=polypus.Aer(), **{name: EXCLUSIVE[name]})
        assert fake.calls == []

    def test_every_conflicting_argument_is_named(self):
        with pytest.raises(
            polypus.ConfigurationError,
            match="target= cannot be combined with infrastructure, backend",
        ):
            _run(target=polypus.Aer(), infrastructure="local", backend="polypus")

    @pytest.mark.parametrize(
        "defaults",
        [
            {"backend": "aer"},
            {"sim_method": "automatic"},
            {"nodes": 1, "cores_per_qpu": 2},
        ],
        ids=repr,
    )
    def test_a_default_value_is_not_a_conflict(self, defaults):
        _run(target=polypus.Aer(), seed=1, **defaults)
