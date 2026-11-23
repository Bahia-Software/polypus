//! The shot sampler's generator, pinned through the public path
//! `polypus_sim::SplitMix64`.
//!
//! Every seeded count of the native backend (contract C-7) is drawn from this
//! stream, so a change to its constants, its step order or its map to `f64`
//! would move every result for a fixed seed. The expected values are
//! independent of this crate: the SplitMix64 reference outputs and the IEEE-754
//! bits of the first draw.

use polypus_sim::SplitMix64;

/// The first outputs of SplitMix64 seeded with `1234567`, from the reference
/// implementation (Vigna, `splitmix64.c`).
#[test]
fn the_stream_matches_the_reference_outputs() {
    let mut rng = SplitMix64::new(1234567);
    let got: Vec<u64> = (0..5).map(|_| rng.next_u64()).collect();
    assert_eq!(
        got,
        [
            6_457_827_717_110_365_317,
            3_203_168_211_198_807_973,
            9_817_491_932_198_370_423,
            4_593_380_528_125_082_431,
            16_408_922_859_458_223_821,
        ]
    );
}

#[test]
fn state_zero_gives_the_reference_first_output() {
    assert_eq!(SplitMix64::new(0).next_u64(), 0xE220_A839_7B1D_CDAF);
}

/// `next_f64` is the top 53 bits of `next_u64` scaled by `2^-53`, exactly.
#[test]
fn next_f64_is_the_top_53_bits_of_next_u64() {
    assert_eq!(
        SplitMix64::new(7).next_f64().to_bits(),
        0x3FD8_F2F8_7916_4C82
    );

    let (mut floats, mut words) = (SplitMix64::new(0x5EED), SplitMix64::new(0x5EED));
    for _ in 0..1000 {
        let expected = (words.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        assert_eq!(floats.next_f64().to_bits(), expected.to_bits());
    }
}
