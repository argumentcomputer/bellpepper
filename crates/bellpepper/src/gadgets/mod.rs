//! Self-contained sub-circuit implementations for various primitives.
pub mod blake2s;
pub use bellpepper_core::boolean;
#[macro_use]
pub mod boolean_utils;
pub mod lookup;
pub mod multieq;
pub mod multipack;
pub use bellpepper_core::num;
pub mod sha256;
pub mod uint32;

use bellpepper_core::SynthesisError;

// TODO: This should probably be removed and we
// should use existing helper methods on `Option`
// for mapping with an error.
/// This basically is just an extension to `Option`
/// which allows for a convenient mapping to an
/// error on `None`.
pub trait Assignment<T> {
    fn get(&self) -> Result<&T, SynthesisError>;
}

impl<T> Assignment<T> for Option<T> {
    fn get(&self) -> Result<&T, SynthesisError> {
        match *self {
            Some(ref v) => Ok(v),
            None => Err(SynthesisError::AssignmentMissing),
        }
    }
}
