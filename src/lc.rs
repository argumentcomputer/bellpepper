use std::ops::{Add, Sub};

use ff::Field;
use pairing::Engine;

use crate::multiexp::DensityTracker;

/// Represents a variable in our constraint system.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Variable(pub(crate) Index);

impl Variable {
    /// This constructs a variable with an arbitrary index.
    /// Circuit implementations are not recommended to use this.
    pub fn new_unchecked(idx: Index) -> Variable {
        Variable(idx)
    }

    /// This returns the index underlying the variable.
    /// Circuit implementations are not recommended to use this.
    pub fn get_unchecked(&self) -> Index {
        self.0
    }
}

/// Represents the index of either an input variable or
/// auxiliary variable.
#[derive(Copy, Clone, PartialEq, Debug, Eq, Hash)]
pub enum Index {
    Input(usize),
    Aux(usize),
}

/// This represents a linear combination of some variables, with coefficients
/// in the scalar field of a pairing-friendly elliptic curve group.
#[derive(Clone)]
pub struct LinearCombination<E: Engine> {
    inputs: Vec<(usize, E::Fr)>,
    aux: Vec<(usize, E::Fr)>,
}

impl<E: Engine> Default for LinearCombination<E> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<E: Engine> LinearCombination<E> {
    pub fn zero() -> LinearCombination<E> {
        LinearCombination {
            inputs: Vec::new(),
            aux: Vec::new(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (Variable, &E::Fr)> + '_ {
        self.inputs
            .iter()
            .map(|(k, v)| (Variable(Index::Input(*k)), v))
            .chain(self.aux.iter().map(|(k, v)| (Variable(Index::Aux(*k)), v)))
    }

    #[inline]
    pub(crate) fn iter_inputs(&self) -> impl Iterator<Item = &(usize, E::Fr)> + '_ {
        self.inputs.iter()
    }

    #[inline]
    pub(crate) fn iter_aux(&self) -> impl Iterator<Item = &(usize, E::Fr)> + '_ {
        self.aux.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Variable, &mut E::Fr)> + '_ {
        self.inputs
            .iter_mut()
            .map(|(k, v)| (Variable(Index::Input(*k)), v))
            .chain(
                self.aux
                    .iter_mut()
                    .map(|(k, v)| (Variable(Index::Aux(*k)), v)),
            )
    }

    fn add_assign_unsimplified_input(&mut self, new_var: usize, coeff: E::Fr) {
        match self
            .inputs
            .binary_search_by_key(&new_var, |(var, _coeff)| *var)
        {
            Ok(index) => {
                self.inputs[index].1 += coeff;
            }
            Err(index) => {
                self.inputs.insert(index, (new_var, coeff));
            }
        }
    }

    fn add_assign_unsimplified_aux(&mut self, new_var: usize, coeff: E::Fr) {
        match self
            .aux
            .binary_search_by_key(&new_var, |(var, _coeff)| *var)
        {
            Ok(index) => {
                self.aux[index].1 += coeff;
            }
            Err(index) => {
                self.aux.insert(index, (new_var, coeff));
            }
        }
    }

    pub fn add_unsimplified(mut self, (coeff, var): (E::Fr, Variable)) -> LinearCombination<E> {
        match var.0 {
            Index::Input(new_var) => {
                self.add_assign_unsimplified_input(new_var, coeff);
            }
            Index::Aux(new_var) => {
                self.add_assign_unsimplified_aux(new_var, coeff);
            }
        }

        self
    }

    fn sub_assign_unsimplified_input(&mut self, new_var: usize, coeff: E::Fr) {
        self.add_assign_unsimplified_input(new_var, -coeff);
    }

    fn sub_assign_unsimplified_aux(&mut self, new_var: usize, coeff: E::Fr) {
        self.add_assign_unsimplified_aux(new_var, -coeff);
    }

    pub fn sub_unsimplified(mut self, (coeff, var): (E::Fr, Variable)) -> LinearCombination<E> {
        match var.0 {
            Index::Input(new_var) => {
                self.sub_assign_unsimplified_input(new_var, coeff);
            }
            Index::Aux(new_var) => {
                self.sub_assign_unsimplified_aux(new_var, coeff);
            }
        }

        self
    }

    pub fn len(&self) -> usize {
        self.inputs.len() + self.aux.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn eval(
        &self,
        mut input_density: Option<&mut DensityTracker>,
        mut aux_density: Option<&mut DensityTracker>,
        input_assignment: &[E::Fr],
        aux_assignment: &[E::Fr],
    ) -> E::Fr {
        let mut acc = E::Fr::zero();

        let one = E::Fr::one();

        for (index, coeff) in self.iter_inputs() {
            let mut tmp = input_assignment[*index];
            if coeff == &one {
                acc += tmp;
            } else {
                tmp *= coeff;
                acc += tmp;
            }

            if let Some(ref mut v) = input_density {
                v.inc(*index);
            }
        }

        for (index, coeff) in self.iter_aux() {
            let mut tmp = aux_assignment[*index];
            if coeff == &one {
                acc += tmp;
            } else {
                tmp *= coeff;
                acc += tmp;
            }

            if let Some(ref mut v) = aux_density {
                v.inc(*index);
            }
        }

        acc
    }
}

impl<E: Engine> Add<(E::Fr, Variable)> for LinearCombination<E> {
    type Output = LinearCombination<E>;

    fn add(self, (coeff, var): (E::Fr, Variable)) -> LinearCombination<E> {
        self.add_unsimplified((coeff, var))
    }
}

impl<E: Engine> Sub<(E::Fr, Variable)> for LinearCombination<E> {
    type Output = LinearCombination<E>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, (coeff, var): (E::Fr, Variable)) -> LinearCombination<E> {
        self.sub_unsimplified((coeff, var))
    }
}

impl<E: Engine> Add<Variable> for LinearCombination<E> {
    type Output = LinearCombination<E>;

    fn add(self, other: Variable) -> LinearCombination<E> {
        self + (E::Fr::one(), other)
    }
}

impl<E: Engine> Sub<Variable> for LinearCombination<E> {
    type Output = LinearCombination<E>;

    fn sub(self, other: Variable) -> LinearCombination<E> {
        self - (E::Fr::one(), other)
    }
}

impl<'a, E: Engine> Add<&'a LinearCombination<E>> for LinearCombination<E> {
    type Output = LinearCombination<E>;

    fn add(mut self, other: &'a LinearCombination<E>) -> LinearCombination<E> {
        for (var, val) in &other.inputs {
            self.add_assign_unsimplified_input(*var, *val);
        }

        for (var, val) in &other.aux {
            self.add_assign_unsimplified_aux(*var, *val);
        }

        self
    }
}

impl<'a, E: Engine> Sub<&'a LinearCombination<E>> for LinearCombination<E> {
    type Output = LinearCombination<E>;

    fn sub(mut self, other: &'a LinearCombination<E>) -> LinearCombination<E> {
        for (var, val) in &other.inputs {
            self.sub_assign_unsimplified_input(*var, *val);
        }

        for (var, val) in &other.aux {
            self.sub_assign_unsimplified_aux(*var, *val);
        }

        self
    }
}

impl<'a, E: Engine> Add<(E::Fr, &'a LinearCombination<E>)> for LinearCombination<E> {
    type Output = LinearCombination<E>;

    fn add(mut self, (coeff, other): (E::Fr, &'a LinearCombination<E>)) -> LinearCombination<E> {
        for (var, val) in &other.inputs {
            self.add_assign_unsimplified_input(*var, *val * coeff);
        }

        for (var, val) in &other.aux {
            self.add_assign_unsimplified_aux(*var, *val * coeff);
        }

        self
    }
}

impl<'a, E: Engine> Sub<(E::Fr, &'a LinearCombination<E>)> for LinearCombination<E> {
    type Output = LinearCombination<E>;

    fn sub(mut self, (coeff, other): (E::Fr, &'a LinearCombination<E>)) -> LinearCombination<E> {
        for (var, val) in &other.inputs {
            self.sub_assign_unsimplified_input(*var, *val * coeff);
        }

        for (var, val) in &other.aux {
            self.sub_assign_unsimplified_aux(*var, *val * coeff);
        }

        self
    }
}

#[cfg(all(test, feature = "groth16"))]
mod tests {
    use super::*;
    use blstrs::Bls12;

    #[test]
    fn test_add_simplify() {
        let n = 5;

        let mut lc = LinearCombination::<Bls12>::zero();

        let mut expected_sums = vec![<Bls12 as Engine>::Fr::zero(); n];
        let mut total_additions = 0;
        for (i, expected_sum) in expected_sums.iter_mut().enumerate() {
            for _ in 0..i + 1 {
                let coeff = <Bls12 as Engine>::Fr::one();
                lc = lc + (coeff, Variable::new_unchecked(Index::Aux(i)));
                *expected_sum += coeff;
                total_additions += 1;
            }
        }

        // There are only as many terms as distinct variable Indexes — not one per addition operation.
        assert_eq!(n, lc.len());
        assert!(lc.len() != total_additions);

        // Each variable has the expected coefficient, the sume of those added by its Index.
        lc.iter().for_each(|(var, coeff)| match var.0 {
            Index::Aux(i) => assert_eq!(expected_sums[i], *coeff),
            _ => panic!("unexpected variable type"),
        });
    }
}
