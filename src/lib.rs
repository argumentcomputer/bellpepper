#![allow(clippy::suspicious_arithmetic_impl)]
//! `bellpepper` is a crate for building zk-SNARK circuits. It provides circuit
//! traits and and primitive structures, as well as basic gadget implementations
//! such as booleans and number abstractions.
//!
//! # Example circuit
//!
//! Say we want to write a circuit that proves we know the preimage to some hash
//! computed using SHA-256d (calling SHA-256 twice). The preimage must have a
//! fixed length known in advance (because the circuit parameters will depend on
//! it), but can otherwise have any value. We take the following strategy:
//!
//! - Witness each bit of the preimage.
//! - Compute `hash = SHA-256d(preimage)` inside the circuit.
//! - Expose `hash` as a public input using multiscalar packing.
//!

#![cfg_attr(all(target_arch = "aarch64", nightly), feature(stdsimd))]

#[cfg(test)]
#[macro_use]
extern crate hex_literal;

pub mod gadgets;
pub mod util_cs;

mod lc;
pub use lc::{Index, LinearCombination, Variable};
mod constraint_system;
pub use constraint_system::{Circuit, ConstraintSystem, Namespace, SynthesisError};

pub const BELLPEPPER_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "groth16")]
pub(crate) fn le_bytes_to_u64s(le_bytes: &[u8]) -> Vec<u64> {
    assert_eq!(
        le_bytes.len() % 8,
        0,
        "length must be divisible by u64 byte length (8-bytes)"
    );
    le_bytes
        .chunks(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}
