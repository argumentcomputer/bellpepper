use crate::gpu::{
    error::{GPUError, GPUResult},
    locks, program, GpuEngine,
};
use ff::Field;
use log::info;
use pairing::Engine;
use rust_gpu_tools::{program_closures, Device, LocalBuffer, Program};
use std::cmp;
use std::ops::MulAssign;

const LOG2_MAX_ELEMENTS: usize = 32; // At most 2^32 elements is supported.
const MAX_LOG2_RADIX: u32 = 8; // Radix256
const MAX_LOG2_LOCAL_WORK_SIZE: u32 = 7; // 128

#[allow(clippy::upper_case_acronyms)]
pub struct FFTKernel<E>
where
    E: Engine + GpuEngine,
{
    program: Program,
    _lock: locks::GPULock, // RFC 1857: struct fields are dropped in the same order as they are declared.
    priority: bool,
    _phantom: std::marker::PhantomData<E>,
}

impl<E> FFTKernel<E>
where
    E: Engine + GpuEngine,
{
    pub fn create(priority: bool) -> GPUResult<FFTKernel<E>> {
        let lock = locks::GPULock::lock();

        // Select the first device for FFT
        let device = *Device::all()
            .first()
            .ok_or(GPUError::Simple("No working GPUs found!"))?;

        let program = program::program::<E>(&device)?;

        info!("FFT: 1 working device(s) selected.");
        info!("FFT: Device 0: {}", device.name());

        Ok(FFTKernel {
            program,
            _lock: lock,
            priority,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Performs FFT on `input`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    pub fn radix_fft(&mut self, input: &mut [E::Fr], omega: &E::Fr, log_n: u32) -> GPUResult<()> {
        let closures = program_closures!(|program, input: &mut [E::Fr]| -> GPUResult<()> {
            let n = 1 << log_n;
            // All usages are safe as the buffers are initialized from either the host or the GPU
            // before they are read.
            let mut src_buffer = unsafe { program.create_buffer::<E::Fr>(n)? };
            let mut dst_buffer = unsafe { program.create_buffer::<E::Fr>(n)? };
            // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
            let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);

            // Precalculate:
            // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
            let mut pq = vec![E::Fr::zero(); 1 << max_deg >> 1];
            let twiddle = omega.pow_vartime([(n >> max_deg) as u64]);
            pq[0] = E::Fr::one();
            if max_deg > 1 {
                pq[1] = twiddle;
                for i in 2..(1 << max_deg >> 1) {
                    pq[i] = pq[i - 1];
                    pq[i].mul_assign(&twiddle);
                }
            }
            let pq_buffer = program.create_buffer_from_slice(&pq)?;

            // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
            let mut omegas = vec![E::Fr::zero(); 32];
            omegas[0] = *omega;
            for i in 1..LOG2_MAX_ELEMENTS {
                omegas[i] = omegas[i - 1].pow_vartime([2u64]);
            }
            let omegas_buffer = program.create_buffer_from_slice(&omegas)?;

            program.write_from_buffer(&mut src_buffer, &*input)?;
            // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
            let mut log_p = 0u32;
            // Each iteration performs a FFT round
            while log_p < log_n {
                // 1=>radix2, 2=>radix4, 3=>radix8, ...
                let deg = cmp::min(max_deg, log_n - log_p);

                if locks::PriorityLock::should_break(self.priority) {
                    return Err(GPUError::GPUTaken);
                }

                let n = 1u32 << log_n;
                let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
                let global_work_size = n >> deg;
                let kernel = program.create_kernel(
                    "radix_fft",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&src_buffer)
                    .arg(&dst_buffer)
                    .arg(&pq_buffer)
                    .arg(&omegas_buffer)
                    .arg(&LocalBuffer::<E::Fr>::new(1 << deg))
                    .arg(&n)
                    .arg(&log_p)
                    .arg(&deg)
                    .arg(&max_deg)
                    .run()?;

                log_p += deg;
                std::mem::swap(&mut src_buffer, &mut dst_buffer);
            }

            program.read_into_buffer(&src_buffer, input)?;

            Ok(())
        });

        self.program.run(closures, input)
    }
}
