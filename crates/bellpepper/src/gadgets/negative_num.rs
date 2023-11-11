use bellpepper_core::{boolean::Boolean, num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::PrimeFieldBits;

/// Allocate Boolean for predicate "num is negative".
/// We have that a number is defined to be negative if the parity bit (the
/// least significant bit) is odd after doubling, meaning that the field element
/// (after doubling) is larger than the underlying prime p that defines the
/// field, then a modular reduction must have been carried out, changing the parity that
/// should be even (since we multiplied by 2) to odd. In other words, we define
/// negative numbers to be those field elements that are larger than p/2.
pub fn allocate_is_negative<F: PrimeFieldBits, CS: ConstraintSystem<F>>(
    mut cs: CS,
    num: &AllocatedNum<F>,
) -> Result<Boolean, SynthesisError> {
    let double_num = num.add(&mut cs.namespace(|| "double num"), num)?;
    let double_num_bits = double_num
        .to_bits_le_strict(&mut cs.namespace(|| "double num bits"))
        .unwrap();

    let lsb_2num = double_num_bits.get(0);
    let num_is_negative = lsb_2num.unwrap();

    Ok(num_is_negative.clone())
}
