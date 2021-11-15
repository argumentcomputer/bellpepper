//! This module contains an [`EvaluationDomain`] abstraction for performing
//! various kinds of polynomial arithmetic on top of the scalar field.
//!
//! In pairing-based SNARKs like [Groth16], we need to calculate a quotient
//! polynomial over a target polynomial with roots at distinct points associated
//! with each constraint of the constraint system. In order to be efficient, we
//! choose these roots to be the powers of a 2<sup>n</sup> root of unity in the
//! field. This allows us to perform polynomial operations in O(n) by performing
//! an O(n log n) FFT over such a domain.
//!
//! [`EvaluationDomain`]: crate::domain::EvaluationDomain
//! [Groth16]: https://eprint.iacr.org/2016/260

#[cfg(any(feature = "cuda", feature = "opencl"))]
use ec_gpu_gen::fft::FftKernel;
use ff::{Field, PrimeField};
use pairing::Engine;

use super::SynthesisError;
use crate::gpu;
use ec_gpu_gen::fft_cpu;
use ec_gpu_gen::threadpool::Worker;

pub struct EvaluationDomain<E: Engine + gpu::GpuEngine> {
    coeffs: Vec<E::Fr>,
    exp: u32,
    omega: E::Fr,
    omegainv: E::Fr,
    geninv: E::Fr,
    minv: E::Fr,
}

impl<E: Engine + gpu::GpuEngine> AsRef<[E::Fr]> for EvaluationDomain<E> {
    fn as_ref(&self) -> &[E::Fr] {
        &self.coeffs
    }
}

impl<E: Engine + gpu::GpuEngine> AsMut<[E::Fr]> for EvaluationDomain<E> {
    fn as_mut(&mut self) -> &mut [E::Fr] {
        &mut self.coeffs
    }
}

impl<E: Engine + gpu::GpuEngine> EvaluationDomain<E> {
    pub fn into_coeffs(self) -> Vec<E::Fr> {
        self.coeffs
    }

    pub fn from_coeffs(mut coeffs: Vec<E::Fr>) -> Result<EvaluationDomain<E>, SynthesisError> {
        // Compute the size of our evaluation domain
        let mut m = 1;
        let mut exp = 0;
        while m < coeffs.len() {
            m *= 2;
            exp += 1;

            // The pairing-friendly curve may not be able to support
            // large enough (radix2) evaluation domains.
            if exp >= E::Fr::S {
                return Err(SynthesisError::PolynomialDegreeTooLarge);
            }
        }
        // Compute omega, the 2^exp primitive root of unity
        let mut omega = E::Fr::root_of_unity();
        for _ in exp..E::Fr::S {
            omega = omega.square();
        }

        // Extend the coeffs vector with zeroes if necessary
        coeffs.resize(m, E::Fr::zero());

        Ok(EvaluationDomain {
            coeffs,
            exp,
            omega,
            omegainv: omega.invert().unwrap(),
            geninv: E::Fr::multiplicative_generator().invert().unwrap(),
            minv: E::Fr::from(m as u64).invert().unwrap(),
        })
    }

    pub fn fft(
        &mut self,
        worker: &Worker,
        kern: &mut Option<gpu::LockedFFTKernel<E>>,
    ) -> gpu::GpuResult<()> {
        best_fft::<E>(
            kern,
            worker,
            &mut [&mut self.coeffs],
            &[self.omega],
            &[self.exp],
        );
        Ok(())
    }

    /// Execute three FFTs in parallel.
    pub fn fft_many(
        domains: &mut [&mut Self],
        worker: &Worker,
        kern: &mut Option<gpu::LockedFFTKernel<E>>,
    ) -> gpu::GpuResult<()> {
        let (mut coeffs, rest): (Vec<_>, Vec<_>) = domains
            .iter_mut()
            .map(|domain| (&mut domain.coeffs[..], (domain.omega, domain.exp)))
            .unzip();
        let (omegas, exps): (Vec<_>, Vec<_>) = rest.into_iter().unzip();
        best_fft(kern, worker, &mut coeffs[..], &omegas, &exps);

        Ok(())
    }

    pub fn ifft(
        &mut self,
        worker: &Worker,
        kern: &mut Option<gpu::LockedFFTKernel<E>>,
    ) -> gpu::GpuResult<()> {
        Self::ifft_many(&mut [self], worker, kern)
    }

    /// Execute multiple IFFTs in parallel.
    pub fn ifft_many(
        domains: &mut [&mut Self],
        worker: &Worker,
        kern: &mut Option<gpu::LockedFFTKernel<E>>,
    ) -> gpu::GpuResult<()> {
        let (mut coeffs, rest): (Vec<_>, Vec<_>) = domains
            .iter_mut()
            .map(|domain| (&mut domain.coeffs[..], (domain.omegainv, domain.exp)))
            .unzip();
        let (omegas, exps): (Vec<_>, Vec<_>) = rest.into_iter().unzip();

        best_fft(kern, worker, &mut coeffs, &omegas, &exps);

        for domain in domains {
            worker.scope(domain.coeffs.len(), |scope, chunk| {
                let minv = domain.minv;

                for v in domain.coeffs.chunks_mut(chunk) {
                    scope.execute(move || {
                        for v in v {
                            *v *= minv;
                        }
                    });
                }
            });
        }

        Ok(())
    }

    pub fn distribute_powers(&mut self, worker: &Worker, g: E::Fr) {
        worker.scope(self.coeffs.len(), |scope, chunk| {
            for (i, v) in self.coeffs.chunks_mut(chunk).enumerate() {
                scope.execute(move || {
                    let mut u = g.pow_vartime(&[(i * chunk) as u64]);
                    for v in v.iter_mut() {
                        *v *= u;
                        u *= g;
                    }
                });
            }
        });
    }

    pub fn coset_fft(
        &mut self,
        worker: &Worker,
        kern: &mut Option<gpu::LockedFFTKernel<E>>,
    ) -> gpu::GpuResult<()> {
        Self::coset_fft_many(&mut [self], worker, kern)
    }

    /// Execute three Coset FFTs in parallel.
    pub fn coset_fft_many(
        domains: &mut [&mut Self],
        worker: &Worker,
        kern: &mut Option<gpu::LockedFFTKernel<E>>,
    ) -> gpu::GpuResult<()> {
        for domain in domains.iter_mut() {
            domain.distribute_powers(worker, E::Fr::multiplicative_generator());
        }

        Self::fft_many(domains, worker, kern)?;

        Ok(())
    }

    pub fn icoset_fft(
        &mut self,
        worker: &Worker,
        kern: &mut Option<gpu::LockedFFTKernel<E>>,
    ) -> gpu::GpuResult<()> {
        let geninv = self.geninv;
        self.ifft(worker, kern)?;
        self.distribute_powers(worker, geninv);
        Ok(())
    }

    /// This evaluates t(tau) for this domain, which is
    /// tau^m - 1 for these radix-2 domains.
    pub fn z(&self, tau: &E::Fr) -> E::Fr {
        let tmp = tau.pow_vartime(&[self.coeffs.len() as u64]);
        tmp - E::Fr::one()
    }

    /// The target polynomial is the zero polynomial in our
    /// evaluation domain, so we must perform division over
    /// a coset.
    pub fn divide_by_z_on_coset(&mut self, worker: &Worker) {
        let i = self.z(&E::Fr::multiplicative_generator()).invert().unwrap();

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for v in self.coeffs.chunks_mut(chunk) {
                scope.execute(move || {
                    for v in v {
                        *v *= i;
                    }
                });
            }
        });
    }

    /// Perform O(n) multiplication of two polynomials in the domain.
    pub fn mul_assign(&mut self, worker: &Worker, other: &EvaluationDomain<E>) {
        assert_eq!(self.coeffs.len(), other.coeffs.len());

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for (a, b) in self
                .coeffs
                .chunks_mut(chunk)
                .zip(other.coeffs.chunks(chunk))
            {
                scope.execute(move || {
                    for (a, b) in a.iter_mut().zip(b.iter()) {
                        *a *= b;
                    }
                });
            }
        });
    }

    /// Perform O(n) subtraction of one polynomial from another in the domain.
    pub fn sub_assign(&mut self, worker: &Worker, other: &EvaluationDomain<E>) {
        assert_eq!(self.coeffs.len(), other.coeffs.len());

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for (a, b) in self
                .coeffs
                .chunks_mut(chunk)
                .zip(other.coeffs.chunks(chunk))
            {
                scope.execute(move || {
                    for (a, b) in a.iter_mut().zip(b.iter()) {
                        *a -= b;
                    }
                });
            }
        });
    }
}

fn best_fft<E: Engine + gpu::GpuEngine>(
    #[allow(unused_variables)] kern: &mut Option<gpu::LockedFFTKernel<E>>,
    worker: &Worker,
    coeffs: &mut [&mut [E::Fr]],
    omegas: &[E::Fr],
    log_ns: &[u32],
) {
    #[cfg(any(feature = "cuda", feature = "opencl"))]
    if let Some(ref mut kern) = kern {
        if kern
            .with(|k: &mut FftKernel<E>| gpu_fft(k, coeffs, omegas, log_ns))
            .is_ok()
        {
            return;
        }
    }

    let log_cpus = worker.log_num_threads();
    for ((a, omega), log_n) in coeffs.iter_mut().zip(omegas.iter()).zip(log_ns.iter()) {
        if *log_n <= log_cpus {
            fft_cpu::serial_fft::<E>(*a, omega, *log_n);
        } else {
            fft_cpu::parallel_fft::<E>(*a, worker, omega, *log_n, log_cpus);
        }
    }
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
pub fn gpu_fft<E: Engine + gpu::GpuEngine>(
    kern: &mut FftKernel<E>,
    coeffs: &mut [&mut [E::Fr]],
    omegas: &[E::Fr],
    log_ns: &[u32],
) -> gpu::GpuResult<()> {
    Ok(kern.radix_fft_many(coeffs, omegas, log_ns)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test multiplying various (low degree) polynomials together and
    // comparing with naive evaluations.
    #[test]
    fn polynomial_arith() {
        use blstrs::Bls12;
        use rand_core::RngCore;

        fn test_mul<E: Engine + gpu::GpuEngine, R: RngCore>(rng: &mut R) {
            let worker = Worker::new();

            for coeffs_a in 0..70 {
                for coeffs_b in 0..70 {
                    let mut a: Vec<_> = (0..coeffs_a).map(|_| E::Fr::random(&mut *rng)).collect();
                    let mut b: Vec<_> = (0..coeffs_b).map(|_| E::Fr::random(&mut *rng)).collect();

                    // naive evaluation
                    let mut naive = vec![E::Fr::zero(); coeffs_a + coeffs_b];
                    for (i1, a) in a.iter().enumerate() {
                        for (i2, b) in b.iter().enumerate() {
                            naive[i1 + i2] += *a * b;
                        }
                    }

                    a.resize(coeffs_a + coeffs_b, E::Fr::zero());
                    b.resize(coeffs_a + coeffs_b, E::Fr::zero());

                    let mut a = EvaluationDomain::<E>::from_coeffs(a).unwrap();
                    let mut b = EvaluationDomain::<E>::from_coeffs(b).unwrap();

                    a.fft(&worker, &mut None).unwrap();
                    b.fft(&worker, &mut None).unwrap();
                    a.mul_assign(&worker, &b);
                    a.ifft(&worker, &mut None).unwrap();

                    for (naive, fft) in naive.iter().zip(a.coeffs.iter()) {
                        assert!(naive == fft);
                    }
                }
            }
        }

        let rng = &mut rand::thread_rng();

        test_mul::<Bls12, _>(rng);
    }

    #[test]
    fn fft_composition() {
        use blstrs::Bls12;
        use pairing::Engine;
        use rand_core::RngCore;

        fn test_comp<E: Engine + gpu::GpuEngine, R: RngCore>(rng: &mut R) {
            let worker = Worker::new();

            for coeffs in 0..10 {
                let coeffs = 1 << coeffs;

                let mut v = vec![];
                for _ in 0..coeffs {
                    v.push(E::Fr::random(&mut *rng));
                }

                let mut domain = EvaluationDomain::<E>::from_coeffs(v.clone()).unwrap();
                domain.ifft(&worker, &mut None).unwrap();
                domain.fft(&worker, &mut None).unwrap();
                assert!(v == domain.coeffs);
                domain.fft(&worker, &mut None).unwrap();
                domain.ifft(&worker, &mut None).unwrap();
                assert!(v == domain.coeffs);
                domain.icoset_fft(&worker, &mut None).unwrap();
                domain.coset_fft(&worker, &mut None).unwrap();
                assert!(v == domain.coeffs);
                domain.coset_fft(&worker, &mut None).unwrap();
                domain.icoset_fft(&worker, &mut None).unwrap();
                assert!(v == domain.coeffs);
            }
        }

        let rng = &mut rand::thread_rng();

        test_comp::<Bls12, _>(rng);
    }
}
