//! `ObservableError` as a `std::error::Error`: the message of every variant,
//! and the source chain of a foreign error carried in `External`.

use std::error::Error;
use std::fmt;

use polypus_observable::ObservableError;

#[test]
fn every_variant_has_a_message_naming_its_data() {
    let msg = ObservableError::BitWidthMismatch {
        num_vars: 5,
        key_len: 3,
    }
    .to_string();
    assert_eq!(
        msg,
        "measurement bitstring of length 3 is too short for an observable over 5 variables"
    );

    let msg = ObservableError::InvalidBitstring("01x".to_string()).to_string();
    assert_eq!(
        msg,
        "measurement bitstring contains a non-binary character: \"01x\""
    );

    let msg = ObservableError::Invalid("num_vars must be >= 1".to_string()).to_string();
    assert_eq!(msg, "invalid observable: num_vars must be >= 1");
}

/// A foreign error, as a non-native observable (the Python callback) boxes it.
#[derive(Debug)]
struct Foreign;

impl fmt::Display for Foreign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "raised by the callback")
    }
}

impl Error for Foreign {}

#[test]
fn an_external_error_keeps_its_message_and_is_the_source() {
    let err = ObservableError::External(Box::new(Foreign));
    // The message is the foreign error's own, verbatim.
    assert_eq!(err.to_string(), "raised by the callback");
    // And the foreign error is reachable through `source`, so a caller can
    // walk the chain down to it.
    let source = err
        .source()
        .expect("External carries its error as the source");
    assert!(source.downcast_ref::<Foreign>().is_some(), "{source:?}");
}

#[test]
fn native_variants_have_no_source() {
    let native = [
        ObservableError::BitWidthMismatch {
            num_vars: 2,
            key_len: 1,
        },
        ObservableError::InvalidBitstring("2".to_string()),
        ObservableError::Invalid("x".to_string()),
    ];
    for err in native {
        assert!(err.source().is_none(), "{err:?}");
    }
}
