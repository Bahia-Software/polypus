//! SHADE and L-SHADE, the success-history adaptive strategies of
//! [`AlgorithmDifferentialEvolution`].
//!
//! Reached through [`AlgorithmDifferentialEvolution::optimize_with_strategy`]
//! with [`DeStrategy::Shade`](crate::DeStrategy::Shade) or
//! [`DeStrategy::LShade`](crate::DeStrategy::LShade). The algorithms are the
//! papers' (R. Tanabe and A. Fukunaga, *Success-History Based Parameter
//! Adaptation for Differential Evolution*, CEC 2013, and *Improving the Search
//! Performance of SHADE Using Linear Population Size Reduction*, CEC 2014),
//! turned into a maximiser and fitted to the parts every DE strategy shares:
//!
//! - angles start uniform in `[0, 2π)`, drawn exactly as DE/rand/1/bin draws
//!   them, so a given seed starts every strategy from the same population,
//!   whose first member an `initial_point` replaces as in DE/rand/1/bin;
//! - a difference between two members is the shortest arc in `(−π, π]`, and a
//!   mutant coordinate is wrapped back into `[0, 2π)`, as DE/rand/1/bin does, instead of the papers' bound repair: the parameters
//!   are angles, so the objective is periodic;
//! - each generation is a single oracle batch, its length checked right away;
//! - the evaluation budget is DE/rand/1/bin's, `population_size ×
//!   (generations + 1)` (see [`Schedule`]), and the run stops early on fitness
//!   stagnation (`tolerance` / `patience`, counted in generations).
//!
//! The constants are the papers' defaults. SHADE: a memory of `H =
//! population_size` slots, an archive as large as the population, and the
//! p-best fraction drawn per individual from `[2/N, 0.2]`. L-SHADE: `H = 6`, an
//! archive of `2.6 N`, `p = 0.11`, the terminal value for `M_CR` and the
//! weighted Lehmer mean for both memories. As in the paper, its population
//! shrinks linearly in the evaluations spent, from `population_size` to 4 when
//! the budget runs out, so it spends the budget in more, smaller generations
//! than `generations` full ones.

use crate::differential_evolution::{
    AlgorithmDifferentialEvolution, AlgorithmDifferentialEvolutionArgs, DeStrategy,
};
use crate::error::OptimizerError;
use crate::outcome::OptimizationOutcome;
use crate::util::{argmax, check_oracle_len, fitness_stagnated, wrap_angle, wrap_difference};
use rand::{Rng, RngExt};
use std::f64::consts::{PI, TAU};

/// The smallest population: the floor of every DE strategy, and the size
/// L-SHADE shrinks towards.
const MIN_POPULATION: usize = AlgorithmDifferentialEvolution::MIN_POPULATION;

/// Standard deviation of the normal distribution `CR` is drawn from, and scale
/// of the Cauchy distribution `F` is drawn from (both papers).
const SPREAD: f64 = 0.1;

/// Value every memory slot starts with, for both `M_CR` and `M_F`.
const INITIAL_MEMORY: f64 = 0.5;

/// The success-history strategy to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Variant {
    /// SHADE (CEC 2013): fixed population.
    Shade,
    /// L-SHADE (CEC 2014): linear population size reduction.
    LShade,
}

impl Variant {
    /// The [`DeStrategy`] this variant runs.
    pub(crate) fn strategy(self) -> DeStrategy {
        match self {
            Variant::Shade => DeStrategy::Shade,
            Variant::LShade => DeStrategy::LShade,
        }
    }

    /// Number of memory slots `H`.
    fn memory_size(self, initial_population: usize) -> usize {
        match self {
            Variant::Shade => initial_population,
            Variant::LShade => 6,
        }
    }

    /// Archive capacity as a multiple of the current population size.
    fn archive_rate(self) -> f64 {
        match self {
            Variant::Shade => 1.0,
            Variant::LShade => 2.6,
        }
    }

    /// The fraction `p` of the population the p-best donor comes from. SHADE
    /// draws it per individual, uniformly between `2/N` and `0.2`; L-SHADE
    /// fixes it at `0.11`.
    fn pbest_fraction<R: Rng>(self, population: usize, rng: &mut R) -> f64 {
        match self {
            Variant::Shade => {
                let low = 2.0 / population as f64;
                low + rng.random::<f64>() * (0.2 - low)
            }
            Variant::LShade => 0.11,
        }
    }
}

/// One trial that strictly improved on its parent: the control parameters
/// that built it and its fitness gain.
struct Success {
    cr: f64,
    f: f64,
    gain: f64,
}

/// The success history: `H` slots of (`M_CR`, `M_F`), written in turn. Each
/// generation that improves at least one member overwrites the next slot.
///
/// Every `M_F` stays in `(0, 1]`: it starts at 0.5 and is only ever replaced
/// by a weighted Lehmer mean of values in `(0, 1]` with non-negative weights
/// that sum to one. That is what guarantees the redraw loop in
/// [`sample`](Self::sample) ends. Every `M_CR` stays in `[0, 1]` the same way.
struct Memory {
    /// `None` is L-SHADE's terminal value: a slot whose successful `CR`s were
    /// all 0 gives `CR = 0` from then on.
    cr: Vec<Option<f64>>,
    f: Vec<f64>,
    next: usize,
}

impl Memory {
    fn new(size: usize) -> Self {
        Self {
            cr: vec![Some(INITIAL_MEMORY); size],
            f: vec![INITIAL_MEMORY; size],
            next: 0,
        }
    }

    /// Draw `(CR, F)` for one individual from a random slot: `CR` from
    /// `N(M_CR, 0.1)` clipped to `[0, 1]` (0 for a terminal slot), `F` from
    /// `Cauchy(M_F, 0.1)`, drawn again while it is not positive and capped at 1.
    fn sample<R: Rng>(&self, rng: &mut R) -> (f64, f64) {
        let slot = rng.random_range(0..self.f.len());
        let cr = match self.cr[slot] {
            Some(mean) => (mean + SPREAD * standard_normal(rng)).clamp(0.0, 1.0),
            None => 0.0,
        };
        let f = loop {
            let f = self.f[slot] + SPREAD * (PI * (rng.random::<f64>() - 0.5)).tan();
            if f > 0.0 {
                break f.min(1.0);
            }
        };
        (cr, f)
    }

    /// Write the generation's successes into the next slot; a generation
    /// without any leaves the memory as it is.
    fn update(&mut self, successes: &[Success], variant: Variant) {
        if successes.is_empty() {
            return;
        }
        let weights = gain_weights(successes);
        let crs: Vec<f64> = successes.iter().map(|s| s.cr).collect();
        let fs: Vec<f64> = successes.iter().map(|s| s.f).collect();
        let k = self.next;
        if let Some(mean) = weighted_lehmer_mean(&fs, &weights) {
            self.f[k] = mean;
        }
        self.cr[k] = match (variant, self.cr[k]) {
            (Variant::Shade, _) => Some(crs.iter().zip(&weights).map(|(cr, w)| w * cr).sum()),
            (Variant::LShade, None) => None,
            (Variant::LShade, Some(_)) if crs.iter().all(|&cr| cr == 0.0) => None,
            (Variant::LShade, Some(previous)) => {
                Some(weighted_lehmer_mean(&crs, &weights).unwrap_or(previous))
            }
        };
        self.next = (k + 1) % self.f.len();
    }
}

/// A standard normal draw (Box–Muller; `rand` alone has no normal
/// distribution and this crate adds no dependency for one). The first uniform
/// is taken from `(0, 1]` so its logarithm is finite.
fn standard_normal<R: Rng>(rng: &mut R) -> f64 {
    let radius = (-2.0 * (1.0 - rng.random::<f64>()).ln()).sqrt();
    radius * (TAU * rng.random::<f64>()).cos()
}

/// The weights `Δfₖ / Σ Δf` of the weighted means. When the gains do not
/// add up to a finite positive number (an oracle returning an infinite
/// fitness), every success weighs the same, so the memory never holds a `NaN`.
fn gain_weights(successes: &[Success]) -> Vec<f64> {
    let total: f64 = successes.iter().map(|s| s.gain).sum();
    if total.is_finite() && total > 0.0 {
        successes.iter().map(|s| s.gain / total).collect()
    } else {
        vec![1.0 / successes.len() as f64; successes.len()]
    }
}

/// The weighted Lehmer mean `Σ w x² / Σ w x`, or `None` when the denominator
/// is not positive.
fn weighted_lehmer_mean(values: &[f64], weights: &[f64]) -> Option<f64> {
    let (squares, sum) = values
        .iter()
        .zip(weights)
        .fold((0.0, 0.0), |(squares, sum), (x, w)| {
            (squares + w * x * x, sum + w * x)
        });
    if sum > 0.0 {
        Some(squares / sum)
    } else {
        None
    }
}

/// Indices of the population from best to worst. `NaN` ranks below every
/// number and ties keep their index order: a total order, so the sort cannot
/// panic and the ranking is reproducible.
fn ranking(fitness: &[f64]) -> Vec<usize> {
    let key = |i: usize| {
        if fitness[i].is_nan() {
            f64::NEG_INFINITY
        } else {
            fitness[i]
        }
    };
    let mut order: Vec<usize> = (0..fitness.len()).collect();
    order.sort_by(|&a, &b| key(b).total_cmp(&key(a)));
    order
}

/// Size of the pool the p-best donor is drawn from: the best
/// `max(2, round(p·n))` members.
fn pbest_pool(fraction: f64, population: usize) -> usize {
    ((fraction * population as f64).round() as usize).clamp(2, population)
}

/// A uniform index in `0..n` other than `i` (`n >= 2`).
fn index_except<R: Rng>(rng: &mut R, n: usize, i: usize) -> usize {
    let r = rng.random_range(0..n - 1);
    if r >= i {
        r + 1
    } else {
        r
    }
}

/// A uniform index in `0..n` other than `a` and `b` (`a != b`, `n >= 3`).
fn index_except_two<R: Rng>(rng: &mut R, n: usize, a: usize, b: usize) -> usize {
    let (low, high) = if a < b { (a, b) } else { (b, a) };
    let mut r = rng.random_range(0..n - 2);
    if r >= low {
        r += 1;
    }
    if r >= high {
        r += 1;
    }
    r
}

/// L-SHADE's population size once `evaluations` of the `budget` are spent:
/// `round(N_init + (N_min - N_init) · evaluations / budget)`, never below
/// [`MIN_POPULATION`].
fn reduced_size(initial: usize, evaluations: u64, budget: u64) -> usize {
    let spent = evaluations as f64 / budget as f64;
    let size = (initial as f64 + (MIN_POPULATION as f64 - initial as f64) * spent).round();
    (size as usize).max(MIN_POPULATION)
}

/// How many members each generation evaluates, and when the run is out of
/// evaluations.
///
/// Every DE strategy has the same budget, `population_size × (generations +
/// 1)` evaluations: the initial population plus `generations` full
/// generations. A generation runs only if its whole batch fits in what is left
/// of it. SHADE keeps its population, so that is exactly `generations`
/// generations. L-SHADE shrinks its population after every generation, so it
/// runs more, smaller ones, until the next would overshoot the budget; it
/// stops less than one (final) population short of it.
///
/// The schedule depends on the arguments alone, never on fitness, so
/// [`DeStrategy::max_generations`](crate::DeStrategy::max_generations) counts a
/// run's generations by stepping the very schedule the optimizer follows.
#[derive(Debug, Clone)]
pub(crate) struct Schedule {
    initial: usize,
    shrinks: bool,
    budget: u64,
    evaluations: u64,
    size: usize,
}

impl Schedule {
    /// The schedule of a run of `population_size` members with the budget of
    /// `generations` generations, once the initial population is evaluated.
    pub(crate) fn new(variant: Variant, population_size: usize, generations: u32) -> Self {
        Self {
            initial: population_size,
            shrinks: variant == Variant::LShade,
            budget: population_size as u64 * (u64::from(generations) + 1),
            evaluations: population_size as u64,
            size: population_size,
        }
    }

    /// Whether the next generation, a batch of the current population, fits in
    /// what is left of the budget.
    pub(crate) fn has_next(&self) -> bool {
        self.size > 0 && self.evaluations + self.size as u64 <= self.budget
    }

    /// Spend one generation of the current population and return the
    /// population size for the next: smaller for L-SHADE, never below
    /// [`MIN_POPULATION`] (nor above the current size).
    pub(crate) fn advance(&mut self) -> usize {
        self.evaluations += self.size as u64;
        if self.shrinks {
            self.size = self
                .size
                .min(reduced_size(self.initial, self.evaluations, self.budget));
        }
        self.size
    }

    /// How many generations the schedule runs.
    pub(crate) fn generations(mut self) -> u64 {
        let mut count = 0;
        while self.has_next() {
            self.advance();
            count += 1;
        }
        count
    }
}

/// Keep the best `size` members, in their current order.
fn keep_best(population: &mut Vec<Vec<f64>>, fitness: &mut Vec<f64>, size: usize) {
    let mut survivors = ranking(fitness);
    survivors.truncate(size);
    survivors.sort_unstable();
    *population = survivors
        .iter()
        .map(|&i| std::mem::take(&mut population[i]))
        .collect();
    *fitness = survivors.iter().map(|&i| fitness[i]).collect();
}

/// Remove random members until the archive holds at most `capacity`, as both
/// papers do when it overflows.
fn trim_archive<R: Rng>(archive: &mut Vec<Vec<f64>>, capacity: usize, rng: &mut R) {
    while archive.len() > capacity {
        let victim = rng.random_range(0..archive.len());
        archive.swap_remove(victim);
    }
}

/// Run SHADE or L-SHADE with the given RNG.
pub(crate) fn run_with_rng<R: Rng>(
    args: AlgorithmDifferentialEvolutionArgs,
    variant: Variant,
    rng: &mut R,
) -> Result<OptimizationOutcome, OptimizerError> {
    let AlgorithmDifferentialEvolutionArgs {
        oracle,
        population_size,
        generations,
        dimensions,
        tolerance,
        patience,
        seed: _,
        initial_point,
        mutation_factor,
        crossover_rate,
    } = args;

    let initial = population_size as usize;
    let dims = dimensions as usize;
    // The same preconditions as DE/rand/1/bin, checked before any RNG draw or
    // oracle call. The population bound also covers what current-to-pbest/1
    // needs: a donor other than the parent and a second one other than both.
    AlgorithmDifferentialEvolution::check_preconditions(
        variant.strategy(),
        population_size,
        dimensions,
        mutation_factor,
        crossover_rate,
        patience,
        initial_point.as_deref(),
    )?;
    let patience = patience as usize;

    // The same draws, in the same order, as DE/rand/1/bin's initialisation, and
    // the same replacement of the first member by the caller's point.
    let mut population: Vec<Vec<f64>> = (0..initial)
        .map(|_| (0..dims).map(|_| rng.random_range(0.0..TAU)).collect())
        .collect();
    if let (Some(point), Some(first)) = (initial_point, population.first_mut()) {
        *first = point.into_iter().map(wrap_angle).collect();
    }
    let mut fitness = oracle.evaluate_batch(&population);
    check_oracle_len(population.len(), fitness.len())?;
    let mut schedule = Schedule::new(variant, initial, generations);

    let mut memory = Memory::new(variant.memory_size(initial));
    let mut archive: Vec<Vec<f64>> = Vec::new();
    let mut best = argmax(&fitness);

    let mut iterations_run = 0usize;
    let mut converged = false;
    let mut fitness_history: Vec<f64> = Vec::new();

    // The population always has the schedule's current size: it starts at
    // `initial` and is cut to each size `advance` returns.
    let mut generation = 0usize;
    while schedule.has_next() {
        iterations_run = generation + 1;
        let size = population.len();
        let order = ranking(&fitness);

        // Build every trial of the generation: current-to-pbest/1 with the
        // archive, then binomial crossover keeping at least one mutant
        // coordinate.
        let mut controls: Vec<(f64, f64)> = Vec::with_capacity(size);
        let mut trials: Vec<Vec<f64>> = Vec::with_capacity(size);
        for i in 0..size {
            let (cr, f) = memory.sample(rng);
            let pool = pbest_pool(variant.pbest_fraction(size, rng), size);
            let pbest = &population[order[rng.random_range(0..pool)]];
            let r1 = index_except(rng, size, i);
            let r2 = index_except_two(rng, size + archive.len(), i, r1);
            let x1 = &population[r1];
            let x2 = if r2 < size {
                &population[r2]
            } else {
                &archive[r2 - size]
            };
            let parent = &population[i];
            let j_rand = if dims > 0 {
                rng.random_range(0..dims)
            } else {
                0
            };
            let trial = (0..dims)
                .map(|j| {
                    if rng.random::<f64>() < cr || j == j_rand {
                        // Angles live on a circle: each difference is the
                        // shortest arc.
                        wrap_angle(
                            parent[j]
                                + f * wrap_difference(pbest[j] - parent[j])
                                + f * wrap_difference(x1[j] - x2[j]),
                        )
                    } else {
                        parent[j]
                    }
                })
                .collect();
            controls.push((cr, f));
            trials.push(trial);
        }

        let trial_fitness = oracle.evaluate_batch(&trials);
        check_oracle_len(trials.len(), trial_fitness.len())?;

        // Selection. A trial at least as good as its parent replaces it (ties
        // go to the trial, as in the papers); only a strict improvement sends
        // the parent to the archive and feeds the memory.
        let mut successes: Vec<Success> = Vec::new();
        for (i, (trial, value)) in trials.into_iter().zip(trial_fitness).enumerate() {
            if value >= fitness[i] {
                let parent = std::mem::replace(&mut population[i], trial);
                if value > fitness[i] {
                    let (cr, f) = controls[i];
                    successes.push(Success {
                        cr,
                        f,
                        gain: value - fitness[i],
                    });
                    archive.push(parent);
                }
                fitness[i] = value;
            }
        }
        memory.update(&successes, variant);

        let target = schedule.advance();
        if target < population.len() {
            keep_best(&mut population, &mut fitness, target);
            log::debug!("Generation {generation}: L-SHADE population reduced to {target}");
        }
        let capacity = (variant.archive_rate() * population.len() as f64).round() as usize;
        trim_archive(&mut archive, capacity, rng);

        // Selection never lowers a member's fitness and the reduction drops the
        // worst members, so the best fitness never falls (contract C-5).
        best = argmax(&fitness);
        fitness_history.push(fitness[best]);
        if fitness_stagnated(&fitness_history, tolerance, patience, generation) {
            converged = true;
            break;
        }
        generation += 1;
    }

    Ok(OptimizationOutcome {
        best_fitness: fitness[best],
        best_params: population.swap_remove(best),
        iterations_run,
        converged,
        fitness_history,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::with_seeded_rng;

    fn success(cr: f64, f: f64, gain: f64) -> Success {
        Success { cr, f, gain }
    }

    #[test]
    fn lehmer_mean_weighs_squares_over_values() {
        // (0.5·0.04 + 0.5·0.36) / (0.5·0.2 + 0.5·0.6) = 0.2 / 0.4 = 0.5.
        let mean = weighted_lehmer_mean(&[0.2, 0.6], &[0.5, 0.5]).unwrap();
        assert!((mean - 0.5).abs() < 1e-12, "{mean}");
        assert_eq!(weighted_lehmer_mean(&[0.0, 0.0], &[0.5, 0.5]), None);
        assert_eq!(weighted_lehmer_mean(&[], &[]), None);
    }

    #[test]
    fn gain_weights_are_proportional_to_the_gains() {
        let weights = gain_weights(&[success(0.5, 0.5, 1.0), success(0.5, 0.5, 3.0)]);
        assert_eq!(weights, vec![0.25, 0.75]);
    }

    #[test]
    fn gain_weights_fall_back_to_equal_for_infinite_gains() {
        let weights = gain_weights(&[success(0.5, 0.5, f64::INFINITY), success(0.5, 0.5, 1.0)]);
        assert_eq!(weights, vec![0.5, 0.5]);
        let overflowing = gain_weights(&[success(0.5, 0.5, f64::MAX), success(0.5, 0.5, f64::MAX)]);
        assert_eq!(overflowing, vec![0.5, 0.5]);
    }

    #[test]
    fn shade_memory_uses_arithmetic_cr_and_lehmer_f() {
        let mut memory = Memory::new(3);
        memory.update(
            &[success(0.2, 0.2, 1.0), success(0.6, 0.6, 1.0)],
            Variant::Shade,
        );
        assert!((memory.cr[0].unwrap() - 0.4).abs() < 1e-12);
        assert!((memory.f[0] - 0.5).abs() < 1e-12);
        assert_eq!(memory.next, 1);
        // The other slots are untouched.
        assert_eq!(memory.cr[1], Some(INITIAL_MEMORY));
        assert_eq!(memory.f[1], INITIAL_MEMORY);
    }

    #[test]
    fn lshade_memory_uses_lehmer_cr() {
        let mut memory = Memory::new(6);
        memory.update(
            &[success(0.2, 0.2, 1.0), success(0.6, 0.6, 1.0)],
            Variant::LShade,
        );
        assert!((memory.cr[0].unwrap() - 0.5).abs() < 1e-12);
        assert!((memory.f[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn lshade_memory_turns_terminal_on_all_zero_cr_and_stays_terminal() {
        let mut memory = Memory::new(2);
        memory.update(&[success(0.0, 0.7, 1.0)], Variant::LShade);
        assert_eq!(memory.cr[0], None);
        // Slot 1 next, then slot 0 again: a terminal slot never comes back.
        memory.update(&[success(0.9, 0.7, 1.0)], Variant::LShade);
        memory.update(&[success(0.9, 0.7, 1.0)], Variant::LShade);
        assert_eq!(memory.cr[0], None);
        assert!(memory.cr[1].is_some());
        // A terminal slot gives CR = 0.
        let draws = with_seeded_rng(Some(3), |rng| {
            (0..50).map(|_| memory.sample(rng)).collect::<Vec<_>>()
        });
        assert!(draws.iter().any(|&(cr, _)| cr == 0.0));
    }

    #[test]
    fn memory_wraps_around_and_skips_empty_generations() {
        let mut memory = Memory::new(2);
        memory.update(&[], Variant::Shade);
        assert_eq!(
            memory.next, 0,
            "a generation without successes writes nothing"
        );
        memory.update(&[success(0.1, 0.1, 1.0)], Variant::Shade);
        memory.update(&[success(0.3, 0.3, 1.0)], Variant::Shade);
        memory.update(&[success(0.9, 0.9, 1.0)], Variant::Shade);
        assert_eq!(memory.next, 1);
        assert!((memory.f[0] - 0.9).abs() < 1e-12);
        assert!((memory.f[1] - 0.3).abs() < 1e-12);
    }

    #[test]
    fn sampled_controls_stay_in_range() {
        // Extreme memories included: F's redraw loop must end and every draw
        // must stay in range.
        let mut memory = Memory::new(3);
        memory.cr = vec![Some(0.0), Some(1.0), None];
        memory.f = vec![f64::MIN_POSITIVE, 1.0, 0.5];
        with_seeded_rng(Some(11), |rng| {
            for _ in 0..10_000 {
                let (cr, f) = memory.sample(rng);
                assert!((0.0..=1.0).contains(&cr), "CR = {cr}");
                assert!(f > 0.0 && f <= 1.0, "F = {f}");
            }
        });
    }

    #[test]
    fn memory_stays_finite_with_infinite_gains() {
        let mut memory = Memory::new(1);
        memory.update(
            &[success(0.3, 0.4, f64::INFINITY), success(0.5, 0.6, 2.0)],
            Variant::LShade,
        );
        assert!(memory.cr[0].unwrap().is_finite());
        assert!(memory.f[0].is_finite() && memory.f[0] > 0.0);
    }

    #[test]
    fn standard_normal_has_unit_moments() {
        let draws: Vec<f64> = with_seeded_rng(Some(5), |rng| {
            (0..20_000).map(|_| standard_normal(rng)).collect()
        });
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let var = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / draws.len() as f64;
        assert!(mean.abs() < 0.03, "mean = {mean}");
        assert!((var - 1.0).abs() < 0.05, "variance = {var}");
        assert!(draws.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn ranking_puts_nan_last_and_keeps_ties_in_order() {
        let fitness = [1.0, f64::NAN, 3.0, 1.0, f64::NEG_INFINITY];
        assert_eq!(ranking(&fitness), vec![2, 0, 3, 1, 4]);
    }

    #[test]
    fn pbest_pool_is_at_least_two_and_at_most_the_population() {
        assert_eq!(pbest_pool(0.11, 4), 2);
        assert_eq!(pbest_pool(0.11, 50), 6);
        assert_eq!(pbest_pool(0.2, 100), 20);
        assert_eq!(pbest_pool(2.0 / 5.0, 5), 2);
        assert_eq!(pbest_pool(1.0, 4), 4);
    }

    #[test]
    fn shade_pbest_fraction_lies_between_two_over_n_and_a_fifth() {
        with_seeded_rng(Some(9), |rng| {
            for _ in 0..1_000 {
                let p = Variant::Shade.pbest_fraction(50, rng);
                assert!((0.04..=0.2).contains(&p), "p = {p}");
                // Below 10 members 2/N exceeds 0.2; the pool is then always 2.
                let small = Variant::Shade.pbest_fraction(5, rng);
                assert_eq!(pbest_pool(small, 5), 2);
            }
        });
        assert_eq!(
            with_seeded_rng(Some(9), |rng| Variant::LShade.pbest_fraction(50, rng)),
            0.11
        );
    }

    #[test]
    fn distinct_indices_cover_every_allowed_value() {
        with_seeded_rng(Some(1), |rng| {
            let mut seen = [0usize; 5];
            for _ in 0..2_000 {
                let r = index_except(rng, 5, 2);
                assert_ne!(r, 2);
                seen[r] += 1;
            }
            assert!(seen.iter().enumerate().all(|(i, &n)| (i == 2) == (n == 0)));

            let mut seen = [0usize; 6];
            for _ in 0..3_000 {
                let r = index_except_two(rng, 6, 4, 1);
                assert!(r != 4 && r != 1);
                seen[r] += 1;
            }
            assert!(seen
                .iter()
                .enumerate()
                .all(|(i, &n)| (i == 1 || i == 4) == (n == 0)));
        });
    }

    #[test]
    fn reduced_size_is_linear_in_evaluations_and_floored_at_four() {
        // 50 members, 100 generations: budget 5050 evaluations.
        assert_eq!(reduced_size(50, 50, 5050), 50);
        assert_eq!(reduced_size(50, 2525, 5050), 27);
        assert_eq!(reduced_size(50, 5050, 5050), 4);
        assert_eq!(reduced_size(4, 44, 44), 4);
        assert_eq!(reduced_size(50, 6000, 5050), 4);
    }

    /// Step a schedule to the end: every generation's size, and the size of
    /// the one that no longer fits.
    fn steps(mut schedule: Schedule) -> (Vec<usize>, usize) {
        let mut sizes = Vec::new();
        while schedule.has_next() {
            sizes.push(schedule.size);
            schedule.advance();
        }
        (sizes, schedule.size)
    }

    #[test]
    fn shade_schedule_runs_exactly_the_given_generations() {
        for (population, generations) in [(4, 0), (4, 1), (10, 25), (50, 100), (1, 7)] {
            let schedule = Schedule::new(Variant::Shade, population, generations);
            assert_eq!(
                schedule.clone().generations(),
                u64::from(generations),
                "{population} × {generations}"
            );
            let (sizes, _) = steps(schedule);
            assert!(sizes.iter().all(|&n| n == population));
        }
    }

    #[test]
    fn lshade_schedule_spends_the_budget_without_overshooting_it() {
        for (population, generations) in [(4, 30), (5, 10), (6, 60), (20, 40), (50, 100), (9, 0)] {
            let budget = (population * (generations as usize + 1)) as u64;
            let schedule = Schedule::new(Variant::LShade, population, generations);
            let count = schedule.clone().generations();
            let (sizes, last) = steps(schedule);
            let spent = (population + sizes.iter().sum::<usize>()) as u64;
            let label = format!("{population} × {generations}: {sizes:?}");
            assert_eq!(sizes.len() as u64, count, "{label}");
            assert!(spent <= budget, "{label}");
            assert!(budget - spent < last as u64, "{label}: {spent} of {budget}");
            assert!(sizes.windows(2).all(|w| w[1] <= w[0]), "{label}");
            assert!(sizes.iter().all(|&n| n >= MIN_POPULATION), "{label}");
            assert!(count >= u64::from(generations), "{label}");
        }
    }

    #[test]
    fn lshade_schedule_with_five_members_drops_to_four_and_runs_longer() {
        // Budget 55: 5 + 5 × 5 members, then 6 generations of 4; 54 spent.
        let (sizes, last) = steps(Schedule::new(Variant::LShade, 5, 10));
        assert_eq!(sizes, [vec![5; 5], vec![4; 6]].concat());
        assert_eq!(last, 4);
    }

    #[test]
    fn an_empty_schedule_ends_at_once() {
        for variant in [Variant::Shade, Variant::LShade] {
            assert_eq!(Schedule::new(variant, 0, 10).generations(), 0);
            assert_eq!(Schedule::new(variant, 8, 0).generations(), 0);
        }
    }

    #[test]
    fn keep_best_drops_the_worst_and_keeps_order() {
        let mut population = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let mut fitness = vec![5.0, f64::NAN, 9.0, 1.0, 7.0];
        keep_best(&mut population, &mut fitness, 3);
        assert_eq!(population, vec![vec![0.0], vec![2.0], vec![4.0]]);
        assert_eq!(fitness, vec![5.0, 9.0, 7.0]);
    }

    #[test]
    fn trim_archive_respects_the_capacity() {
        let mut archive: Vec<Vec<f64>> = (0..10).map(|i| vec![f64::from(i)]).collect();
        with_seeded_rng(Some(2), |rng| trim_archive(&mut archive, 4, rng));
        assert_eq!(archive.len(), 4);
        with_seeded_rng(Some(2), |rng| trim_archive(&mut archive, 6, rng));
        assert_eq!(archive.len(), 4, "an archive under capacity is left alone");
    }
}
