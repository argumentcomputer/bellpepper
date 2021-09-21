use std::env;

#[cfg(feature = "opencl")]
use ec_gpu_gen::Limb64;
use log::info;
#[cfg(feature = "cuda")]
use rust_gpu_tools::cuda;
#[cfg(feature = "opencl")]
use rust_gpu_tools::opencl;
use rust_gpu_tools::{Device, Framework, GPUError as GpuToolsError, Program};

#[cfg(not(all(feature = "cuda", feature = "opencl")))]
use crate::gpu::error::GPUError;
use crate::gpu::error::GPUResult;
#[cfg(feature = "opencl")]
use crate::gpu::sources;
use crate::gpu::GpuEngine;
use pairing::Engine;

/// Returns the program for the preferred [`rust_gpu_tools::device::Framework`].
///
/// If the device supports CUDA, then CUDA is used, else OpenCL. You can force a selection with
/// the environment variable `BELLMAN_GPU_FRAMEWORK`, which can be set either to `cuda` or `opencl`.
pub fn program<E>(device: &Device) -> GPUResult<Program>
where
    E: Engine + GpuEngine,
{
    let framework = match env::var("BELLMAN_GPU_FRAMEWORK") {
        Ok(env) => match env.as_ref() {
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    Framework::Cuda
                }

                #[cfg(not(feature = "cuda"))]
                return Err(GPUError::Simple("CUDA framework is not supported, please compile with the `cuda` feature enabled."));
            }
            "opencl" => {
                #[cfg(feature = "opencl")]
                {
                    Framework::Opencl
                }

                #[cfg(not(feature = "opencl"))]
                return Err(GPUError::Simple("OpenCL framework is not supported, please compile with the `opencl` feature enabled."));
            }
            _ => device.framework(),
        },
        Err(_) => device.framework(),
    };
    program_use_framework::<E>(device, &framework)
}

/// Returns the program for the specified [`rust_gpu_tools::device::Framework`].
pub fn program_use_framework<E>(device: &Device, framework: &Framework) -> GPUResult<Program>
where
    E: Engine + GpuEngine,
{
    match framework {
        #[cfg(feature = "cuda")]
        Framework::Cuda => {
            info!("Using kernel on CUDA.");
            let kernel = include_bytes!(env!("CUDA_MULTIEXP_FATBIN"));
            let cuda_device = device.cuda_device().ok_or(GpuToolsError::DeviceNotFound)?;
            let program = cuda::Program::from_bytes(cuda_device, kernel)?;
            Ok(Program::Cuda(program))
        }
        #[cfg(feature = "opencl")]
        Framework::Opencl => {
            info!("Using kernel on OpenCL.");
            let src = sources::kernel::<E, Limb64>();
            let opencl_device = device
                .opencl_device()
                .ok_or(GpuToolsError::DeviceNotFound)?;
            let program = opencl::Program::from_opencl(opencl_device, &src)?;
            Ok(Program::Opencl(program))
        }
    }
}
