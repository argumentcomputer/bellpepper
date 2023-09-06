#![deny(missing_debug_implementations)]
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

mod lc;
pub use lc::{Index, LinearCombination, Variable};
mod constraint_system;
pub use constraint_system::{Circuit, ConstraintSystem, Namespace, SynthesisError};
mod gadgets;
pub use gadgets::{boolean, num};
mod util_cs;
pub use util_cs::{test_cs, Comparable, Constraint, Delta};

pub const BELLPEPPER_VERSION: &str = env!("CARGO_PKG_VERSION");
