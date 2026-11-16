//! Quantum kernels: the fidelity kernel `K(x, y) = |⟨φ(x)|φ(y)⟩|²` of a feature
//! map made of encoders, the Gram matrix a kernel method (an SVM with a
//! precomputed kernel) trains on.
//!
//! A [`FeatureMap`] is an encoder stack compiled for a feature count: the
//! circuit `U(x)` that prepares `|φ(x)⟩ = U(x)|0…0⟩`, built by the same layers,
//! with the same checks, as the encoder part of a
//! [`CompiledModel`](crate::CompiledModel)'s template. A [`FidelityKernel`]
//! pairs the samples of one or two sets and gives, for every pair `(i, j)`,
//! either
//!
//! - the **compute-uncompute** circuit `U(x_i)` then `U(y_j)†` then
//!   `measure_all`, whose probability of `0…0` is `K(x_i, y_j)` (read from
//!   counts with [`FidelityKernel::fidelity_from_counts`], an unbiased
//!   estimator, or from exact probabilities), or
//! - the unmeasured state circuits `U(x_i)` and `U(y_j)` themselves, for a
//!   caller that holds statevectors and takes the inner product directly
//!   (`N + M` simulations instead of `N · M` circuits).
//!
//! Like the rest of the crate it executes nothing.
//!
//! ## Pairs and symmetry
//!
//! With two sets the pairs are every `(i, j)`, row-major: `n1 · n2` of them.
//! With one set (`y = x`, a training Gram matrix) only the upper triangle is
//! computed, **diagonal included** — `n · (n + 1) / 2` pairs, row-major — and
//! [`FidelityKernel::matrix`] mirrors it, so the matrix is symmetric bit for
//! bit. The diagonal is executed like any other pair, not set to `1`: on noisy
//! hardware `K(x, x) < 1` is information. Two sets that happen to be equal are
//! not detected: every entry is computed.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::OnceLock;

use polypus_circuit::{ConcreteCircuit, GateInstruction};

use crate::error::{QmlError, ValidationError};
use crate::layers::Layer;
use crate::model::{emit_layers, plan_layers, LayerAllocation};

/// An encoder stack compiled for `num_features` features: the circuit `U(x)`
/// whose state `U(x)|0…0⟩` is a sample's feature vector in a quantum kernel.
///
/// Every layer is planned exactly as [`QuantumModel::compile`] plans it, so
/// the same stack raises the same errors (an amplitude encoder that is not
/// first, more features than the encoders can carry, …) and emits the same
/// gates: [`circuit`](Self::circuit) is, gate for gate, what a model's
/// template emits for these layers, before any trainable layer, basis change
/// or measurement.
///
/// [`QuantumModel::compile`]: crate::QuantumModel::compile
#[derive(Debug, Clone)]
pub struct FeatureMap {
    num_qubits: usize,
    layers: Vec<Layer>,
    num_features: usize,
    allocations: Vec<LayerAllocation>,
}

impl FeatureMap {
    /// Compile `layers` over `num_qubits` qubits for samples of `num_features`
    /// features.
    ///
    /// # Errors
    ///
    /// In order: [`ValidationError::NoQubits`], [`ValidationError::EmptyModel`]
    /// for no layer, each layer's own check (as in
    /// [`QuantumModel::compile`](crate::QuantumModel::compile)), then
    /// [`ValidationError::TrainableLayerInFeatureMap`] for the first layer that
    /// reserves trainable parameters: the feature map of a kernel is made of
    /// encoders only.
    pub fn compile(
        num_qubits: usize,
        layers: Vec<Layer>,
        num_features: usize,
    ) -> Result<Self, ValidationError> {
        if num_qubits < 1 {
            return Err(ValidationError::NoQubits);
        }
        if layers.is_empty() {
            return Err(ValidationError::EmptyModel);
        }
        let (allocations, _) = plan_layers(num_qubits, &layers, num_features)?;
        if let Some(layer) = allocations
            .iter()
            .position(|alloc| !alloc.params.is_empty())
        {
            return Err(ValidationError::TrainableLayerInFeatureMap { layer });
        }
        Ok(FeatureMap {
            num_qubits,
            layers,
            num_features,
            allocations,
        })
    }

    /// The qubits of `U(x)`, the width of the counts a kernel circuit gives.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// The features per sample it was compiled for.
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// `U(x)`: the encoders' gates for the sample `x`, every angle fixed, with
    /// no measurement.
    ///
    /// # Errors
    ///
    /// [`QmlError::FeatureCountMismatch`] for a sample of another width, and
    /// what the encoders reject in a sample ([`QmlError::ZeroNormSample`] for
    /// amplitude encoding, [`QmlError::NonBinaryFeature`] for basis encoding).
    pub fn circuit(&self, x: &[f64]) -> Result<ConcreteCircuit, QmlError> {
        if x.len() != self.num_features {
            return Err(QmlError::FeatureCountMismatch {
                expected: self.num_features,
                got: x.len(),
            });
        }
        let (qc, _) = emit_layers(self.num_qubits, &self.layers, &self.allocations, x)?;
        Ok(qc.assign_parameters(&[])?)
    }

    /// [`circuit`](Self::circuit) of every sample, in order: the states a
    /// [`FidelityKernel`] pairs.
    ///
    /// # Errors
    ///
    /// The first sample's error that [`circuit`](Self::circuit) gives.
    pub fn states<S: AsRef<[f64]>>(&self, samples: &[S]) -> Result<FeatureStates, QmlError> {
        let circuits = samples
            .iter()
            .map(|sample| self.circuit(sample.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(FeatureStates {
            num_qubits: self.num_qubits,
            circuits,
        })
    }
}

/// The fidelity kernel of a [`FeatureMap`] over the samples `x` (rows) and `y`
/// (columns), or over `x` alone: the pairs to compute, their circuits, how to
/// read a pair's results, and how the pairs assemble into the matrix (see the
/// module docs).
///
/// Every sample is encoded at construction, so every check on the samples runs
/// before any circuit is executed. Each column's `U(y)†` is built once, the
/// first time a pair's circuit is asked for ([`circuits`](Self::circuits)): a
/// caller that takes inner products of the states never pays for it. Pairs are
/// numbered `0..num_pairs()` row-major (only `j >= i` for a symmetric kernel)
/// and can be taken in blocks ([`circuits`](Self::circuits)), the results of
/// the blocks concatenating in pair order.
#[derive(Debug, Clone)]
pub struct FidelityKernel {
    num_qubits: usize,
    /// `U(x_i)`, unmeasured.
    rows: Vec<ConcreteCircuit>,
    /// `U(y_j)`, unmeasured; `None` for a symmetric kernel, whose columns are
    /// its rows.
    columns: Option<Vec<ConcreteCircuit>>,
    /// `U(y_j)†` (of the rows for a symmetric kernel), built by the first
    /// [`circuits`](Self::circuits) call.
    uncompute: OnceLock<Vec<ConcreteCircuit>>,
    num_pairs: usize,
}

impl FidelityKernel {
    /// The kernel of `feature_map` between the samples `x` and `y`, or, with
    /// `y` `None`, the symmetric kernel of `x` with itself:
    /// [`from_states`](Self::from_states) over [`FeatureMap::states`] of each.
    ///
    /// # Errors
    ///
    /// - [`QmlError::KernelTooLarge`] when the matrix's entries do not fit in
    ///   memory's address space, checked before any sample is encoded;
    /// - a sample [`FeatureMap::circuit`] rejects, `x`'s first, then `y`'s.
    pub fn new<S: AsRef<[f64]>>(
        feature_map: &FeatureMap,
        x: &[S],
        y: Option<&[S]>,
    ) -> Result<Self, QmlError> {
        kernel_size(x.len(), y.map(<[S]>::len))?;
        let rows = feature_map.states(x)?;
        let columns = y.map(|y| feature_map.states(y)).transpose()?;
        Self::from_states(rows, columns)
    }

    /// The kernel between the states `rows` and `columns`, or, with `columns`
    /// `None`, the symmetric kernel of `rows` with themselves.
    ///
    /// # Errors
    ///
    /// - [`QmlError::KernelTooLarge`] when the matrix's entries do not fit in
    ///   memory's address space;
    /// - [`QmlError::StateWidthMismatch`] for states of feature maps of
    ///   different widths.
    pub fn from_states(
        rows: FeatureStates,
        columns: Option<FeatureStates>,
    ) -> Result<Self, QmlError> {
        let num_pairs = kernel_size(rows.len(), columns.as_ref().map(FeatureStates::len))?;
        if let Some(columns) = &columns {
            if columns.num_qubits != rows.num_qubits {
                return Err(QmlError::StateWidthMismatch {
                    rows: rows.num_qubits,
                    columns: columns.num_qubits,
                });
            }
        }
        let num_qubits = rows.num_qubits;
        let rows = rows.circuits;
        let columns = columns.map(|columns| columns.circuits);
        Ok(FidelityKernel {
            num_qubits,
            rows,
            columns,
            uncompute: OnceLock::new(),
            num_pairs,
        })
    }

    /// The samples of `x`: the matrix's rows.
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// The samples of `y`, or of `x` for a symmetric kernel: the matrix's
    /// columns.
    pub fn num_columns(&self) -> usize {
        self.columns.as_ref().map_or(self.rows.len(), Vec::len)
    }

    /// Whether this is the symmetric kernel of one set with itself, whose
    /// pairs are the upper triangle.
    pub fn is_symmetric(&self) -> bool {
        self.columns.is_none()
    }

    /// The pairs to compute: `rows · columns`, or `n · (n + 1) / 2` for a
    /// symmetric kernel. One circuit each.
    pub fn num_pairs(&self) -> usize {
        self.num_pairs
    }

    /// The qubits of every circuit, and the width of their counts' keys.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// The unmeasured `U(x_i)` of every row, in row order.
    pub fn row_states(&self) -> &[ConcreteCircuit] {
        &self.rows
    }

    /// The unmeasured `U(y_j)` of every column, in column order (the rows', for
    /// a symmetric kernel).
    pub fn column_states(&self) -> &[ConcreteCircuit] {
        self.columns.as_deref().unwrap_or(&self.rows)
    }

    /// The `(row, column)` of every pair in `pairs`, in pair order, or `None`
    /// if `pairs` reaches past [`num_pairs`](Self::num_pairs).
    pub fn pairs(&self, pairs: Range<usize>) -> Option<Vec<(usize, usize)>> {
        if pairs.start > pairs.end || pairs.end > self.num_pairs {
            return None;
        }
        let num_columns = self.num_columns();
        let mut out = Vec::with_capacity(pairs.len());
        if pairs.is_empty() {
            return Some(out);
        }
        // The first pair, then walk row-major: along the row, then to the next
        // row's first column (the diagonal, for a symmetric kernel).
        let (mut i, mut j) = match self.columns {
            Some(_) => (pairs.start / num_columns, pairs.start % num_columns),
            None => {
                let (mut i, mut offset) = (0, 0);
                // Row `i` holds the `n - i` pairs `(i, i..n)`.
                while offset + (num_columns - i) <= pairs.start {
                    offset += num_columns - i;
                    i += 1;
                }
                (i, i + (pairs.start - offset))
            }
        };
        for _ in pairs {
            out.push((i, j));
            j += 1;
            if j == num_columns {
                i += 1;
                j = if self.columns.is_some() { 0 } else { i };
            }
        }
        Some(out)
    }

    /// The compute-uncompute circuit of every pair in `pairs`, in pair order:
    /// `U(x_i)`, then `U(y_j)†`, then one `measure_all`, whose probability of
    /// `0…0` is the pair's kernel value. `None` if `pairs` reaches past
    /// [`num_pairs`](Self::num_pairs).
    ///
    /// The first call builds every column's `U(y_j)†`, once for the kernel.
    ///
    /// # Errors
    ///
    /// [`QmlError::Circuit`] if a circuit could not be inverted or assembled
    /// (neither can happen to an encoder's states, which hold no measurement,
    /// no declared-gate call and no free angle).
    pub fn circuits(&self, pairs: Range<usize>) -> Option<Result<Vec<ConcreteCircuit>, QmlError>> {
        let pairs = self.pairs(pairs)?;
        let uncompute = match self.uncompute() {
            Ok(uncompute) => uncompute,
            Err(e) => return Some(Err(e)),
        };
        Some(
            pairs
                .into_iter()
                .map(|(i, j)| self.pair_circuit(i, &uncompute[j]))
                .collect(),
        )
    }

    /// Every column's `U(y_j)†`, built on the first call. Two threads that race
    /// to build it build the same circuits; one is kept.
    fn uncompute(&self) -> Result<&[ConcreteCircuit], QmlError> {
        if let Some(built) = self.uncompute.get() {
            return Ok(built);
        }
        let built = self
            .column_states()
            .iter()
            .map(ConcreteCircuit::inverse)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.uncompute.get_or_init(|| built))
    }

    fn pair_circuit(
        &self,
        i: usize,
        uncompute: &ConcreteCircuit,
    ) -> Result<ConcreteCircuit, QmlError> {
        let compute = &self.rows[i];
        let mut gates = Vec::with_capacity(compute.gates().len() + uncompute.gates().len() + 1);
        gates.extend_from_slice(compute.gates());
        gates.extend_from_slice(uncompute.gates());
        gates.push(GateInstruction::MeasureAll);
        Ok(ConcreteCircuit::try_new(self.num_qubits, gates)?)
    }

    /// A pair's kernel value from the counts of its circuit: the frequency of
    /// `0…0` (an absent key is `0.0`), over the total of the counts. An
    /// unbiased estimator of `K`, always in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// [`QmlError::EmptyCounts`] for no counts or a zero total;
    /// [`QmlError::CountsWidthMismatch`] for a key that is not
    /// [`num_qubits`](Self::num_qubits) bits wide (C-3).
    pub fn fidelity_from_counts(&self, counts: &HashMap<String, u64>) -> Result<f64, QmlError> {
        self.check_keys(counts.keys())?;
        // Exact in any order while the total stays below 2^53, which a circuit's
        // counts always do: they total its `shots`, a `u32`.
        let total: f64 = counts.values().map(|&c| c as f64).sum();
        if total == 0.0 {
            return Err(QmlError::EmptyCounts);
        }
        let zeros = counts.get(&self.zeros()).map_or(0.0, |&c| c as f64);
        Ok(zeros / total)
    }

    /// A pair's kernel value from the exact probabilities of its circuit: the
    /// probability of `0…0` (an absent key is `0.0`), clamped to `[0, 1]`
    /// (rounding can take `|⟨φ|φ⟩|²` a few ulps past 1, a value no fidelity
    /// has).
    ///
    /// # Errors
    ///
    /// Those of [`fidelity_from_counts`](Self::fidelity_from_counts): an empty
    /// map is [`QmlError::EmptyCounts`].
    pub fn fidelity_from_probabilities(
        &self,
        probabilities: &HashMap<String, f64>,
    ) -> Result<f64, QmlError> {
        self.check_keys(probabilities.keys())?;
        if probabilities.is_empty() {
            return Err(QmlError::EmptyCounts);
        }
        let zeros = probabilities.get(&self.zeros()).copied().unwrap_or(0.0);
        Ok(zeros.clamp(0.0, 1.0))
    }

    /// The `num_qubits`-bit all-zeros key (C-3).
    fn zeros(&self) -> String {
        "0".repeat(self.num_qubits)
    }

    fn check_keys<'a>(&self, keys: impl Iterator<Item = &'a String>) -> Result<(), QmlError> {
        for key in keys {
            if key.len() != self.num_qubits {
                return Err(QmlError::CountsWidthMismatch {
                    expected: self.num_qubits,
                    got: key.len(),
                });
            }
        }
        Ok(())
    }

    /// The `rows × columns` matrix, row-major, from one value per pair in pair
    /// order (all [`num_pairs`](Self::num_pairs) of them). A symmetric
    /// kernel's lower triangle is the mirror of its upper one, so
    /// `K[j][i]` and `K[i][j]` are the same `f64`.
    ///
    /// # Errors
    ///
    /// [`QmlError::CountsLengthMismatch`] for another number of values;
    /// [`QmlError::KernelTooLarge`] if the matrix cannot be allocated.
    pub fn matrix(&self, values: &[f64]) -> Result<Vec<f64>, QmlError> {
        if values.len() != self.num_pairs {
            return Err(QmlError::CountsLengthMismatch {
                expected: self.num_pairs,
                got: values.len(),
            });
        }
        let mut matrix = self.zero_matrix()?;
        self.place(&mut matrix, 0, values)?;
        Ok(matrix)
    }

    /// The `rows × columns` matrix, row-major, with every entry `0.0`: what
    /// [`place`](Self::place) fills block by block, so that a caller never
    /// holds the pairs' values and the matrix at once.
    ///
    /// # Errors
    ///
    /// [`QmlError::KernelTooLarge`] if the matrix cannot be allocated.
    pub fn zero_matrix(&self) -> Result<Vec<f64>, QmlError> {
        let (n1, n2) = (self.num_rows(), self.num_columns());
        let entries = n1 * n2; // checked at construction
        let mut matrix = Vec::new();
        matrix
            .try_reserve_exact(entries)
            .map_err(|_| QmlError::KernelTooLarge {
                rows: n1,
                columns: n2,
            })?;
        matrix.resize(entries, 0.0);
        Ok(matrix)
    }

    /// Write the values of the pairs `first..first + values.len()`, in pair
    /// order, at their entries of `matrix` (row-major, `rows × columns`, as
    /// [`zero_matrix`](Self::zero_matrix) gives it), and for a symmetric
    /// kernel at their mirrors too, the same `f64`. Every pair placed once, in
    /// blocks of any size, gives [`matrix`](Self::matrix) of all the values,
    /// bit for bit.
    ///
    /// # Errors
    ///
    /// [`QmlError::CountsLengthMismatch`] if `matrix` does not have
    /// `rows × columns` entries, or if the pairs reach past
    /// [`num_pairs`](Self::num_pairs) (`got` is then the end they reach).
    pub fn place(&self, matrix: &mut [f64], first: usize, values: &[f64]) -> Result<(), QmlError> {
        let (n1, n2) = (self.num_rows(), self.num_columns());
        if matrix.len() != n1 * n2 {
            return Err(QmlError::CountsLengthMismatch {
                expected: n1 * n2,
                got: matrix.len(),
            });
        }
        let past = QmlError::CountsLengthMismatch {
            expected: self.num_pairs,
            got: first.saturating_add(values.len()),
        };
        let end = first
            .checked_add(values.len())
            .filter(|&end| end <= self.num_pairs)
            .ok_or_else(|| past.clone())?;
        if self.columns.is_some() {
            // The pairs of a full kernel are its entries, in order.
            matrix[first..end].copy_from_slice(values);
            return Ok(());
        }
        let pairs = self.pairs(first..end).ok_or(past)?;
        for (&(i, j), &value) in pairs.iter().zip(values) {
            matrix[i * n2 + j] = value;
            matrix[j * n2 + i] = value;
        }
        Ok(())
    }
}

/// The pairs of a kernel over `rows` samples and `columns` of them (`None`:
/// the symmetric kernel of the rows), once its `rows · columns` f64 entries
/// are known to be addressable, or [`QmlError::KernelTooLarge`].
fn kernel_size(rows: usize, columns: Option<usize>) -> Result<usize, QmlError> {
    let num_columns = columns.unwrap_or(rows);
    let entries = rows
        .checked_mul(num_columns)
        .filter(|&n| {
            n.checked_mul(std::mem::size_of::<f64>())
                .is_some_and(|bytes| bytes <= isize::MAX as usize)
        })
        .ok_or(QmlError::KernelTooLarge {
            rows,
            columns: num_columns,
        })?;
    Ok(match columns {
        Some(_) => entries,
        // `n² · 8 <= isize::MAX`, so `n² + n` cannot overflow.
        None => rows * (rows + 1) / 2,
    })
}

/// The states of some samples under one [`FeatureMap`]: `U(x)` for each, in
/// sample order, from [`FeatureMap::states`] — so every sample has passed the
/// encoders' checks. What [`FidelityKernel::from_states`] pairs.
#[derive(Debug, Clone)]
pub struct FeatureStates {
    num_qubits: usize,
    circuits: Vec<ConcreteCircuit>,
}

impl FeatureStates {
    /// How many samples.
    pub fn len(&self) -> usize {
        self.circuits.len()
    }

    /// Whether there are no samples.
    pub fn is_empty(&self) -> bool {
        self.circuits.is_empty()
    }

    /// The qubits of the feature map that encoded them.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// `U(x)` of each sample, unmeasured.
    pub fn circuits(&self) -> &[ConcreteCircuit] {
        &self.circuits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{AngleEncoder, IqpEncoder, RotationAxis};

    fn angle_map(num_qubits: usize, num_features: usize) -> FeatureMap {
        FeatureMap::compile(
            num_qubits,
            vec![Layer::AngleEncoder(AngleEncoder::new(RotationAxis::Ry))],
            num_features,
        )
        .unwrap()
    }

    fn rows(n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|i| vec![0.1 * i as f64]).collect()
    }

    #[test]
    fn pairs_are_row_major_and_the_triangle_includes_the_diagonal() {
        let map = angle_map(1, 1);
        let full = FidelityKernel::new(&map, &rows(2), Some(&rows(3))).unwrap();
        assert_eq!(full.num_pairs(), 6);
        assert_eq!(
            full.pairs(0..6).unwrap(),
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        );
        assert_eq!(full.pairs(2..5).unwrap(), [(0, 2), (1, 0), (1, 1)]);

        let symmetric = FidelityKernel::new(&map, &rows(4), None).unwrap();
        assert_eq!(symmetric.num_pairs(), 10);
        let all = symmetric.pairs(0..10).unwrap();
        assert_eq!(
            all,
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 2),
                (2, 3),
                (3, 3)
            ]
        );
        // Any block starts where the whole enumeration has it.
        for start in 0..10 {
            for end in start..=10 {
                assert_eq!(
                    symmetric.pairs(start..end).unwrap(),
                    all[start..end],
                    "{start}..{end}"
                );
            }
        }
        assert!(symmetric.pairs(3..11).is_none());
        assert!(symmetric.circuits(10..11).is_none());
        assert_eq!(symmetric.pairs(10..10).unwrap(), []);
    }

    #[test]
    fn the_triangle_count_is_n_times_n_plus_one_over_two() {
        let map = angle_map(1, 1);
        for n in 1..12 {
            let kernel = FidelityKernel::new(&map, &rows(n), None).unwrap();
            assert_eq!(kernel.num_pairs(), n * (n + 1) / 2, "n = {n}");
            assert_eq!(
                kernel.pairs(0..kernel.num_pairs()).unwrap().len(),
                kernel.num_pairs()
            );
        }
    }

    #[test]
    fn a_pair_circuit_is_compute_then_uncompute_then_one_measure_all() {
        let map = angle_map(2, 2);
        let x = [vec![0.3, -0.4]];
        let y = [vec![1.1, 0.2]];
        let kernel = FidelityKernel::new(&map, &x, Some(&y)).unwrap();
        let circuit = kernel.circuits(0..1).unwrap().unwrap().remove(0);
        let ry = |qubit, theta| GateInstruction::Ry {
            qubit,
            theta: polypus_circuit::Fixed(theta),
        };
        assert_eq!(
            circuit.gates(),
            [
                ry(0, 0.3),
                ry(1, -0.4),
                ry(1, -0.2),
                ry(0, -1.1),
                GateInstruction::MeasureAll
            ]
        );
        assert_eq!(circuit.num_qubits(), 2);
        // The state circuits are the encoders' gates, unmeasured.
        assert_eq!(kernel.row_states()[0].gates(), [ry(0, 0.3), ry(1, -0.4)]);
        assert_eq!(kernel.column_states()[0].gates(), [ry(0, 1.1), ry(1, 0.2)]);
    }

    /// `U(y)†` is built by the first request for a pair's circuit, not at
    /// construction: the exact route, which takes inner products of the
    /// states, never builds it. A kernel stays `Send + Sync` and clones.
    #[test]
    fn the_uncompute_circuits_are_built_on_the_first_request() {
        fn shareable<T: Send + Sync + Clone + std::fmt::Debug>(_: &T) {}
        let map = angle_map(1, 1);
        let kernel = FidelityKernel::new(&map, &rows(3), Some(&rows(2))).unwrap();
        shareable(&kernel);
        assert!(kernel.uncompute.get().is_none());
        assert_eq!(kernel.column_states().len(), 2);
        let _ = kernel.pairs(0..6).unwrap();
        assert!(kernel.uncompute.get().is_none());
        let first = kernel.circuits(0..2).unwrap().unwrap();
        assert_eq!(kernel.uncompute.get().map(Vec::len), Some(2));
        // Built once: later blocks reuse it, and a clone carries it.
        let built = kernel.uncompute.get().unwrap().as_ptr();
        let later = kernel.circuits(2..6).unwrap().unwrap();
        assert_eq!(kernel.uncompute.get().unwrap().as_ptr(), built);
        assert_eq!(first.len() + later.len(), 6);
        assert_eq!(kernel.clone().uncompute.get().map(Vec::len), Some(2));
    }

    #[test]
    fn the_symmetric_kernel_uncomputes_its_own_rows() {
        let map = angle_map(1, 1);
        let kernel = FidelityKernel::new(&map, &[vec![0.5], vec![0.9]], None).unwrap();
        // Pair 1 is (0, 1): U(x_0) then U(x_1)†.
        let circuit = kernel.circuits(1..2).unwrap().unwrap().remove(0);
        let ry = |theta| GateInstruction::Ry {
            qubit: 0,
            theta: polypus_circuit::Fixed(theta),
        };
        assert_eq!(
            circuit.gates(),
            [ry(0.5), ry(-0.9), GateInstruction::MeasureAll]
        );
        assert!(kernel.is_symmetric());
        assert_eq!(kernel.column_states(), kernel.row_states());
    }

    fn counts(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|&(k, v)| (k.to_string(), v)).collect()
    }

    #[test]
    fn the_counts_reader_takes_the_all_zeros_frequency() {
        let map = angle_map(3, 1);
        let kernel = FidelityKernel::new(&map, &rows(1), None).unwrap();
        assert_eq!(
            kernel
                .fidelity_from_counts(&counts(&[("000", 30), ("010", 70)]))
                .unwrap(),
            0.3
        );
        // An absent key is zero, not an error and never NaN.
        assert_eq!(
            kernel.fidelity_from_counts(&counts(&[("100", 5)])).unwrap(),
            0.0
        );
        assert_eq!(
            kernel.fidelity_from_counts(&HashMap::new()),
            Err(QmlError::EmptyCounts)
        );
        assert_eq!(
            kernel.fidelity_from_counts(&counts(&[("000", 0)])),
            Err(QmlError::EmptyCounts)
        );
        assert_eq!(
            kernel.fidelity_from_counts(&counts(&[("00", 8)])),
            Err(QmlError::CountsWidthMismatch {
                expected: 3,
                got: 2
            })
        );
    }

    #[test]
    fn the_probabilities_reader_takes_and_clamps_the_all_zeros_probability() {
        let map = angle_map(2, 1);
        let kernel = FidelityKernel::new(&map, &rows(1), None).unwrap();
        let probabilities = |pairs: &[(&str, f64)]| -> HashMap<String, f64> {
            pairs.iter().map(|&(k, v)| (k.to_string(), v)).collect()
        };
        assert_eq!(
            kernel
                .fidelity_from_probabilities(&probabilities(&[("00", 0.25), ("11", 0.75)]))
                .unwrap(),
            0.25
        );
        assert_eq!(
            kernel
                .fidelity_from_probabilities(&probabilities(&[("01", 1.0)]))
                .unwrap(),
            0.0
        );
        let past_one = 1.0 + 4.0 * f64::EPSILON;
        assert_eq!(
            kernel
                .fidelity_from_probabilities(&probabilities(&[("00", past_one)]))
                .unwrap(),
            1.0
        );
        assert_eq!(
            kernel.fidelity_from_probabilities(&HashMap::new()),
            Err(QmlError::EmptyCounts)
        );
        assert_eq!(
            kernel.fidelity_from_probabilities(&probabilities(&[("000", 1.0)])),
            Err(QmlError::CountsWidthMismatch {
                expected: 2,
                got: 3
            })
        );
    }

    #[test]
    fn the_matrix_mirrors_the_triangle_bit_for_bit() {
        let map = angle_map(1, 1);
        let kernel = FidelityKernel::new(&map, &rows(3), None).unwrap();
        // Pair order: (0,0) (0,1) (0,2) (1,1) (1,2) (2,2).
        let values = [1.0, 0.1, 0.2, 0.9, 0.3, 0.8];
        assert_eq!(
            kernel.matrix(&values).unwrap(),
            [1.0, 0.1, 0.2, 0.1, 0.9, 0.3, 0.2, 0.3, 0.8]
        );
        assert_eq!(
            kernel.matrix(&values[..5]),
            Err(QmlError::CountsLengthMismatch {
                expected: 6,
                got: 5
            })
        );
        let full = FidelityKernel::new(&map, &rows(2), Some(&rows(3))).unwrap();
        let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        assert_eq!(full.matrix(&values).unwrap(), values);
    }

    /// Placing the values block by block, in blocks of any size, fills the
    /// same matrix as `matrix` of them all, bit for bit, a symmetric one
    /// mirrored; and a block past the pairs, or a matrix of another size, is
    /// refused.
    #[test]
    fn placing_blocks_fills_the_matrix_of_all_the_values() {
        let map = angle_map(1, 1);
        let symmetric = FidelityKernel::new(&map, &rows(4), None).unwrap();
        let full = FidelityKernel::new(&map, &rows(3), Some(&rows(4))).unwrap();
        for kernel in [&symmetric, &full] {
            let values: Vec<f64> = (0..kernel.num_pairs())
                .map(|k| 0.05 + 0.07 * k as f64)
                .collect();
            let whole = kernel.matrix(&values).unwrap();
            for block in [1, 2, 3, 5, kernel.num_pairs()] {
                let mut matrix = kernel.zero_matrix().unwrap();
                for (b, chunk) in values.chunks(block).enumerate() {
                    kernel.place(&mut matrix, b * block, chunk).unwrap();
                }
                assert!(
                    matrix
                        .iter()
                        .zip(&whole)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "block {block}"
                );
            }
            let mut matrix = kernel.zero_matrix().unwrap();
            let n = kernel.num_pairs();
            assert_eq!(
                kernel.place(&mut matrix, n - 1, &[0.5, 0.5]),
                Err(QmlError::CountsLengthMismatch {
                    expected: n,
                    got: n + 1
                })
            );
            assert_eq!(
                kernel.place(&mut matrix[1..], 0, &[0.5]),
                Err(QmlError::CountsLengthMismatch {
                    expected: matrix.len(),
                    got: matrix.len() - 1
                })
            );
            assert!(kernel.place(&mut matrix, n, &[]).is_ok());
        }
    }

    #[test]
    fn the_feature_map_reuses_the_layer_checks() {
        let amplitude_second = vec![
            Layer::AngleEncoder(AngleEncoder::new(RotationAxis::Ry)),
            Layer::AmplitudeEncoder(crate::AmplitudeEncoder),
        ];
        assert_eq!(
            FeatureMap::compile(2, amplitude_second, 2).unwrap_err(),
            ValidationError::AmplitudeEncoderNotFirst
        );
        assert_eq!(
            FeatureMap::compile(2, vec![Layer::AmplitudeEncoder(crate::AmplitudeEncoder)], 5)
                .unwrap_err(),
            ValidationError::TooManyFeatures { max: 4, got: 5 }
        );
        assert_eq!(
            FeatureMap::compile(2, vec![Layer::Iqp(IqpEncoder::new())], 3).unwrap_err(),
            ValidationError::NotEnoughQubits {
                needed: 3,
                active: 2
            }
        );
        assert_eq!(
            FeatureMap::compile(0, vec![Layer::Iqp(IqpEncoder::new())], 1).unwrap_err(),
            ValidationError::NoQubits
        );
        assert_eq!(
            FeatureMap::compile(2, Vec::new(), 1).unwrap_err(),
            ValidationError::EmptyModel
        );
        let trainable = vec![
            Layer::Iqp(IqpEncoder::new()),
            Layer::HardwareEfficient(crate::HardwareEfficientAnsatz::new(1)),
        ];
        assert_eq!(
            FeatureMap::compile(2, trainable, 2).unwrap_err(),
            ValidationError::TrainableLayerInFeatureMap { layer: 1 }
        );
    }

    #[test]
    fn samples_are_checked_at_construction_x_before_y() {
        let map = angle_map(2, 2);
        let good = [vec![0.1, 0.2]];
        let wide = [vec![0.1, 0.2, 0.3]];
        assert_eq!(
            FidelityKernel::new(&map, &wide, Some(&good)).unwrap_err(),
            QmlError::FeatureCountMismatch {
                expected: 2,
                got: 3
            }
        );
        assert_eq!(
            FidelityKernel::new(&map, &good, Some(&wide[..])).unwrap_err(),
            QmlError::FeatureCountMismatch {
                expected: 2,
                got: 3
            }
        );
        let amplitude =
            FeatureMap::compile(1, vec![Layer::AmplitudeEncoder(crate::AmplitudeEncoder)], 2)
                .unwrap();
        assert_eq!(
            FidelityKernel::new(&amplitude, &[vec![0.0, 0.0]], None).unwrap_err(),
            QmlError::ZeroNormSample
        );
        let basis = FeatureMap::compile(2, vec![Layer::Basis(crate::BasisEncoder)], 2).unwrap();
        assert_eq!(
            FidelityKernel::new(&basis, &[vec![1.0, 0.5]], None).unwrap_err(),
            QmlError::NonBinaryFeature {
                feature: 1,
                got: 0.5
            }
        );
    }

    #[test]
    fn states_of_feature_maps_of_different_widths_do_not_pair() {
        let (narrow, wide) = (angle_map(1, 1), angle_map(2, 1));
        let x = [vec![0.1]];
        let states = |map: &FeatureMap| map.states(&x).unwrap();
        assert_eq!(
            FidelityKernel::from_states(states(&narrow), Some(states(&wide))).unwrap_err(),
            QmlError::StateWidthMismatch {
                rows: 1,
                columns: 2
            }
        );
        // Two maps of one width do pair: the kernel between their states.
        let other = FeatureMap::compile(
            1,
            vec![Layer::AngleEncoder(AngleEncoder::new(RotationAxis::Rx))],
            1,
        )
        .unwrap();
        let kernel = FidelityKernel::from_states(states(&narrow), Some(states(&other))).unwrap();
        assert_eq!(kernel.num_pairs(), 1);
        assert_eq!(states(&narrow).len(), 1);
        assert!(!states(&narrow).is_empty());
        assert_eq!(states(&wide).num_qubits(), 2);
    }

    #[test]
    fn a_kernel_whose_matrix_does_not_fit_is_a_typed_error() {
        // A matrix of 2^66 entries (2^34 on a 32-bit target) does not fit in the
        // address space. The samples are zero-sized, so the test allocates
        // nothing: the size is checked before any sample is encoded.
        let map = angle_map(1, 0);
        let n = 1usize << (usize::BITS / 2 + 1);
        let x = vec![[0.0f64; 0]; n];
        assert_eq!(
            FidelityKernel::new(&map, &x, None).unwrap_err(),
            QmlError::KernelTooLarge {
                rows: n,
                columns: n
            }
        );
        // Entries that fit in a `usize` but whose bytes do not.
        let rows_only = vec![[0.0f64; 0]; 1usize << (usize::BITS / 2 - 1)];
        let columns = [[0.0f64; 0]; 1usize << (usize::BITS / 2 - 1)];
        assert!(matches!(
            FidelityKernel::new(&map, &rows_only, Some(&columns[..])),
            Err(QmlError::KernelTooLarge { .. })
        ));
    }
}
