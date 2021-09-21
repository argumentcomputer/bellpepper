mod error;

pub use self::error::*;

#[cfg(any(feature = "cuda", feature = "opencl"))]
mod locks;

#[cfg(any(feature = "cuda", feature = "opencl"))]
pub use self::locks::*;

#[cfg(any(feature = "cuda", feature = "opencl"))]
mod program;

#[cfg(any(feature = "cuda", feature = "opencl"))]
mod sources;

#[cfg(any(feature = "cuda", feature = "opencl"))]
pub use self::sources::*;

#[cfg(any(feature = "cuda", feature = "opencl"))]
mod utils;

#[cfg(any(feature = "cuda", feature = "opencl"))]
pub use self::utils::*;

#[cfg(any(feature = "cuda", feature = "opencl"))]
mod fft;

#[cfg(any(feature = "cuda", feature = "opencl"))]
pub use self::fft::*;

#[cfg(any(feature = "cuda", feature = "opencl"))]
mod multiexp;

#[cfg(any(feature = "cuda", feature = "opencl"))]
pub use self::multiexp::*;

#[cfg(not(any(feature = "cuda", feature = "opencl")))]
mod nogpu;

#[cfg(not(any(feature = "cuda", feature = "opencl")))]
pub use self::nogpu::*;

#[cfg(any(feature = "cuda", feature = "opencl"))]
pub use ec_gpu::GpuEngine;
#[cfg(not(any(feature = "cuda", feature = "opencl")))]
pub trait GpuEngine {}
#[cfg(not(any(feature = "cuda", feature = "opencl")))]
impl<E: pairing::Engine> GpuEngine for E {}
