use ff::PrimeField;
use groupy::CurveAffine;
use rayon::prelude::*;

use crate::bls::{Engine, PairingCurveAffine};
use crate::groth16::multiscalar::*;
use crate::SynthesisError;

/// Returns the miller loop evaluated on inputs, i.e.
/// e(l_1,r_1)e(l_2,r_2)...
/// NOTE: the result is not in the final subgroup, one must run
/// `E::final_exponentiation` to use the final result.
pub(crate) fn pairing_miller_affine<E: Engine>(
    left: &[E::G1Affine],
    right: &[E::G2Affine],
) -> Result<E::Fqk, SynthesisError> {
    if left.len() != right.len() {
        return Err(SynthesisError::IncompatibleLengthVector(
            "pairing_miller_affine left and right".to_string(),
        ));
    }
    let pairs = left
        .par_iter()
        .map(|e| e.prepare())
        .zip(right.par_iter().map(|e| e.prepare()))
        .collect::<Vec<_>>();
    let pairs_ref: Vec<_> = pairs.iter().map(|(a, b)| (a, b)).collect();

    Ok(E::miller_loop(pairs_ref.iter()))
}

/// Returns the miller loop result of the inner pairing product
pub(crate) fn pairing<E: Engine>(
    left: &[E::G1Affine],
    right: &[E::G2Affine],
) -> Result<E::Fqk, SynthesisError> {
    E::final_exponentiation(&pairing_miller_affine::<E>(left, right)?)
        .ok_or(SynthesisError::InvalidPairing)
}

pub(crate) fn multiexponentiation<G: CurveAffine>(
    left: &[G],
    right: &[G::Scalar],
) -> Result<G::Projective, SynthesisError> {
    if left.len() != right.len() {
        return Err(SynthesisError::IncompatibleLengthVector(
            "multiexponentiation left and right".to_string(),
        ));
    }

    let table = precompute_fixed_window::<G>(&left, WINDOW_SIZE);
    let getter = |i: usize| -> <G::Scalar as PrimeField>::Repr { right[i].into_repr() };
    Ok(par_multiscalar::<_, G>(
        &ScalarList::Getter(getter, right.len()),
        &table,
        std::mem::size_of::<<G::Scalar as PrimeField>::Repr>() * 8,
    ))
}
