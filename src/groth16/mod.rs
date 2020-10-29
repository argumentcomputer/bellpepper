//! The [Groth16] proving system.
//!
//! [Groth16]: https://eprint.iacr.org/2016/260

#[cfg(test)]
mod tests;

mod ext;
mod generator;
mod mapped_params;
mod params;
mod proof;
mod prover;
mod verifier;
mod verifying_key;

mod multiscalar;

pub use self::ext::*;
pub use self::generator::*;
pub use self::mapped_params::*;
pub use self::params::*;
pub use self::proof::*;
pub use self::prover::*;
pub use self::verifier::*;
pub use self::verifying_key::*;
