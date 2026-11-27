//! `polypus_sim::SplitMix64` is a public path: the generator
//! `Statevector::sample` draws shots from, re-exported from
//! `polypus_circuit::rng`, where its stream is pinned. This checks that the
//! path exists and names that same type.

#[test]
fn the_public_path_is_the_workspace_generator() {
    // Compiles only if both paths name one type.
    let mut rng: polypus_circuit::rng::SplitMix64 = polypus_sim::SplitMix64::new(7);
    let _ = rng.next_u64();
}
