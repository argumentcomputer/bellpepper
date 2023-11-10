use bellpepper_core::{
    boolean::{AllocatedBit, Boolean},
    num::Num,
    ConstraintSystem, SynthesisError,
};
use ff::PrimeField;

// Returns a Boolean which is true if any of its arguments are true.
#[macro_export]
macro_rules! or {
    ($cs:expr, $a:expr, $b:expr) => {
        Boolean::or(
            bellpepper_core::ConstraintSystem::namespace(&mut $cs, || format!("{} or {}", stringify!($a), stringify!($b))),
            $a,
            $b,
        )
    };
    ($cs:expr, $a:expr, $b:expr, $c:expr, $($x:expr),+) => {{
        let or_tmp_cs_ =  bellpepper_core::ConstraintSystem::namespace(&mut $cs, || format!("or({})", stringify!(vec![$a, $b, $c, $($x),*])));
        super::or_v(or_tmp_cs_, &[$a, $b, $c, $($x),*])
    }};
    ($cs:expr, $a:expr, $($x:expr),+) => {{
        let mut or_tmp_cs_ =  bellpepper_core::ConstraintSystem::namespace(&mut $cs, || format!("or {}", stringify!(vec![$a, $($x),*])));
        let or_tmp_ = or!(or_tmp_cs_, $($x),*)?;
        or!(or_tmp_cs_, $a, &or_tmp_)
    }};
}

// Returns a Boolean which is true if all of its arguments are true.
#[macro_export]
macro_rules! and {
    ($cs:expr, $a:expr, $b:expr) => {
        Boolean::and(
            bellpepper_core::ConstraintSystem::namespace(&mut $cs, || format!("{} and {}", stringify!($a), stringify!($b))),
            $a,
            $b,
        )
    };
    ($cs:expr, $a:expr, $b:expr, $c:expr, $($x:expr),+) => {{
        let and_tmp_cs_ = bellpepper_core::ConstraintSystem::namespace(&mut $cs, || format!("and({})", stringify!([$a, $b, $c, $($x),*])));
        super::and_v(and_tmp_cs_, &[$a, $b, $c, $($x),*])
    }};
    ($cs:ident, $a:expr, $($x:expr),+) => {{
        let mut and_tmp_cs_ =  bellpepper_core::ConstraintSystem::namespace(&mut $cs, || format!("and({})", stringify!([$a, $($x),*])));
        let and_tmp_ = and!(and_tmp_cs_, $($x),*)?;
        and!(and_tmp_cs_, $a, &and_tmp_)
    }};

}

// Allocate a Boolean which is true if and only if `num` is zero.
fn alloc_num_is_zero<CS: ConstraintSystem<F>, F: PrimeField>(
    mut cs: CS,
    num: &Num<F>,
) -> Result<Boolean, SynthesisError> {
    let num_value = num.get_value();
    let x = num_value.unwrap_or(F::ZERO);
    let is_zero = num_value.map(|n| n == F::ZERO);

    // result = (x == 0)
    let result = AllocatedBit::alloc(cs.namespace(|| "x = 0"), is_zero)?;

    // result * x = 0
    // This means that at least one of result or x is zero.
    cs.enforce(
        || "result or x is 0",
        |lc| lc + result.get_variable(),
        |_| num.lc(F::ONE),
        |lc| lc,
    );

    // Inverse of `x`, if it exists, otherwise one.
    let q = cs.alloc(
        || "q",
        || {
            let tmp = x.invert();
            if tmp.is_some().into() {
                Ok(tmp.unwrap())
            } else {
                Ok(F::ONE)
            }
        },
    )?;

    // (x + result) * q = 1.
    // This enforces that x and result are not both 0.
    cs.enforce(
        || "(x + result) * q = 1",
        |_| num.lc(F::ONE) + result.get_variable(),
        |lc| lc + q,
        |lc| lc + CS::one(),
    );

    // Taken together, these constraints enforce that exactly one of `x` and `result` is 0.
    // Since result is constrained to be boolean, that means `result` is true iff `x` is 0.

    Ok(Boolean::Is(result))
}

/// Variadic or.
pub fn or_v<CS: ConstraintSystem<F>, F: PrimeField>(
    cs: CS,
    v: &[&Boolean],
) -> Result<Boolean, SynthesisError> {
    assert!(
        v.len() >= 4,
        "with less than 4 elements, or_v is more expensive than repeated or"
    );

    or_v_unchecked_for_optimization(cs, v)
}

/// Unchecked variadic or.
pub fn or_v_unchecked_for_optimization<CS: ConstraintSystem<F>, F: PrimeField>(
    mut cs: CS,
    v: &[&Boolean],
) -> Result<Boolean, SynthesisError> {
    // Count the number of true values in v.
    let count_true = v.iter().fold(Num::zero(), |acc, b| {
        acc.add_bool_with_coeff(CS::one(), b, F::ONE)
    });

    // If the number of true values is zero, then none of the values is true.
    // Therefore, nor(v0, v1, ..., vn) is true.
    let nor = alloc_num_is_zero(&mut cs.namespace(|| "nor"), &count_true)?;

    Ok(nor.not())
}

/// Variadic and.
pub fn and_v<CS: ConstraintSystem<F>, F: PrimeField>(
    mut cs: CS,
    v: &[&Boolean],
) -> Result<Boolean, SynthesisError> {
    assert!(
        v.len() >= 4,
        "with less than 4 elements, and_v is more expensive than repeated and"
    );

    // Count the number of false values in v.
    let count_false = v.iter().fold(Num::zero(), |acc, b| {
        acc.add_bool_with_coeff(CS::one(), &b.not(), F::ONE)
    });

    // If the number of false values is zero, then all of the values are true.
    // Therefore, and(v0, v1, ..., vn) is true.
    let and = alloc_num_is_zero(&mut cs.namespace(|| "nor_of_nots"), &count_false)?;

    Ok(and)
}

#[cfg(test)]
mod tests {
    use bellpepper_core::{boolean::Boolean, test_cs::TestConstraintSystem};
    use blstrs::Scalar as Fr;
    use proptest::prelude::*;

    proptest! {
        #[test]
        // needs to return Result because the macros use ?.
        fn test_and_or_v((x0, x1, x2, x3, x4) in any::<(bool, bool, bool, bool, bool)>()) {
            let mut cs = TestConstraintSystem::<Fr>::new();

            let a = Boolean::Constant(x0);
            let b = Boolean::Constant(x1);
            let c = Boolean::Constant(x2);
            let d = Boolean::Constant(x3);
            let e = Boolean::Constant(x4);

            let and0 = and!(cs, &a, &b, &c).unwrap();
            let and1 = and!(cs, &a, &b, &c, &d).unwrap();
            let and2 = and!(cs, &a, &b, &c, &d, &e).unwrap();

            let or0 = or!(cs, &a, &b, &c).unwrap();
            let or1 = or!(cs, &a, &b, &c, &d).unwrap();
            let or2 = or!(cs, &a, &b, &c, &d, &e).unwrap();

            let expected_and0 = x0 && x1 && x2;
            let expected_and1 = x0 && x1 && x2 && x3;
            let expected_and2 = x0 && x1 && x2 && x3 && x4;

            let expected_or0 = x0 || x1 || x2;
            let expected_or1 = x0 || x1 || x2 || x3;
            let expected_or2 = x0 || x1 || x2 || x3 || x4;

            assert_eq!(expected_and0, and0.get_value().unwrap());
            assert_eq!(expected_and1, and1.get_value().unwrap());
            assert_eq!(expected_and2, and2.get_value().unwrap());
            assert_eq!(expected_or0, or0.get_value().unwrap());
            assert_eq!(expected_or1, or1.get_value().unwrap());
            assert_eq!(expected_or2, or2.get_value().unwrap());
            assert!(cs.is_satisfied());
        }
    }
}
