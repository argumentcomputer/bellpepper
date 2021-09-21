#![allow(clippy::upper_case_acronyms)]

#[cfg(any(feature = "cuda", feature = "opencl"))]
use rust_gpu_tools::GPUError as GpuToolsError;

#[derive(thiserror::Error, Debug)]
pub enum GPUError {
    #[error("GPUError: {0}")]
    Simple(&'static str),
    #[cfg(any(feature = "cuda", feature = "opencl"))]
    #[error("GPU tools error: {0}")]
    GpuTools(#[from] GpuToolsError),
    #[cfg(any(feature = "cuda", feature = "opencl"))]
    #[error("GPU taken by a high priority process!")]
    GPUTaken,
    #[cfg(any(feature = "cuda", feature = "opencl"))]
    #[error("No kernel is initialized!")]
    KernelUninitialized,
    #[error("GPU accelerator is disabled!")]
    GPUDisabled,
}

pub type GPUResult<T> = std::result::Result<T, GPUError>;

#[cfg(any(feature = "cuda", feature = "opencl"))]
impl From<std::boxed::Box<dyn std::any::Any + std::marker::Send>> for GPUError {
    fn from(e: std::boxed::Box<dyn std::any::Any + std::marker::Send>) -> Self {
        match e.downcast::<Self>() {
            Ok(err) => *err,
            Err(_) => GPUError::Simple("An unknown GPU error happened!"),
        }
    }
}
