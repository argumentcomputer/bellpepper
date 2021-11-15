use super::error::{GpuError, GpuResult};
use ec_gpu_gen::threadpool::Worker;
use ff::PrimeField;
use group::prime::PrimeCurveAffine;
use std::marker::PhantomData;
use std::sync::Arc;

pub struct MultiexpKernel<E>(PhantomData<E>)
where
    E: Engine;

impl<E> MultiexpKernel<E>
where
    E: Engine,
{
    pub fn create(_: bool) -> GpuResult<Self> {
        Err(GpuError::GpuDisabled)
    }

    pub fn multiexp<G>(
        &mut self,
        _: &Worker,
        _: Arc<Vec<G>>,
        _: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
        _: usize,
        _: usize,
    ) -> GpuResult<<G as PrimeCurveAffine>::Curve>
    where
        G: PrimeCurveAffine,
    {
        Err(GpuError::GpuDisabled)
    }
}

use pairing::Engine;

macro_rules! locked_kernel {
    ($class:ident) => {
        #[allow(clippy::upper_case_acronyms)]
        pub struct $class<E>(PhantomData<E>);

        impl<E> $class<E>
        where
            E: Engine,
        {
            pub fn new(_: bool) -> $class<E> {
                $class::<E>(PhantomData)
            }

            pub fn with<F, R, K>(&mut self, _: F) -> GpuResult<R>
            where
                F: FnMut(&mut K) -> GpuResult<R>,
            {
                return Err(GpuError::GpuDisabled);
            }
        }
    };
}

locked_kernel!(LockedFFTKernel);
locked_kernel!(LockedMultiexpKernel);
