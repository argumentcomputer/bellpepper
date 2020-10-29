// --prove                  Benchmark prover
// --verify                 Benchmark verifier
// --proofs <num>           Sets number of proofs in a batch
// --public <num>           Sets number of public inputs
// --private <num>          Sets number of private inputs
// --gpu                    Enables GPU
// --samples                Number of runs
// --dummy                  Skip param generation and generate dummy params/proofs
use std::sync::Arc;
use std::time::Instant;

use bellperson::groth16::{
    create_random_proof_batch, generate_random_parameters, prepare_verifying_key,
    verify_proofs_batch, Parameters, Proof, VerifyingKey,
};
use bellperson::{
    bls::{Bls12, Engine, Fr},
    Circuit, ConstraintSystem, SynthesisError,
};
use fff::{Field, PrimeField, ScalarEngine};
use groupy::CurveProjective;
use rand::{thread_rng, Rng};
use structopt::StructOpt;

macro_rules! timer {
    ($e:expr) => {{
        let before = Instant::now();
        let ret = $e;
        (
            ret,
            (before.elapsed().as_secs() * 1000 as u64 + before.elapsed().subsec_millis() as u64),
        )
    }};
}

#[derive(Clone)]
pub struct DummyDemo {
    pub public: usize,
    pub private: usize,
}

impl<E: Engine> Circuit<E> for DummyDemo {
    fn synthesize<CS: ConstraintSystem<E>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        assert!(self.public >= 1);
        let mut x_val = E::Fr::from_str("2");
        let mut x = cs.alloc_input(|| "", || x_val.ok_or(SynthesisError::AssignmentMissing))?;
        let mut pubs = 1;

        for _ in 0..self.private + self.public - 1 {
            // Allocate: x * x = x2
            let x2_val = x_val.map(|mut e| {
                e.square();
                e
            });

            let x2 = if pubs < self.public {
                pubs += 1;
                cs.alloc_input(|| "", || x2_val.ok_or(SynthesisError::AssignmentMissing))?
            } else {
                cs.alloc(|| "", || x2_val.ok_or(SynthesisError::AssignmentMissing))?
            };

            // Enforce: x * x = x2
            cs.enforce(|| "", |lc| lc + x, |lc| lc + x, |lc| lc + x2);

            x = x2;
            x_val = x2_val;
        }

        cs.enforce(
            || "",
            |lc| lc + (x_val.unwrap(), CS::one()),
            |lc| lc + CS::one(),
            |lc| lc + x,
        );

        Ok(())
    }
}

fn random_points<C: CurveProjective, R: Rng>(count: usize, rng: &mut R) -> Vec<C::Affine> {
    // Number of distinct points is limited because generating random points is very time
    // consuming, so it's better to just repeat them.
    const DISTINT_POINTS: usize = 100;
    (0..DISTINT_POINTS)
        .map(|_| C::random(rng).into_affine())
        .collect::<Vec<_>>()
        .into_iter()
        .cycle()
        .take(count)
        .collect()
}

fn dummy_proofs<E: Engine, R: Rng>(count: usize, rng: &mut R) -> Vec<Proof<E>> {
    (0..count)
        .map(|_| Proof {
            a: E::G1::random(rng).into_affine(),
            b: E::G2::random(rng).into_affine(),
            c: E::G1::random(rng).into_affine(),
        })
        .collect()
}

fn dummy_inputs<E: Engine, R: Rng>(count: usize, rng: &mut R) -> Vec<<E as ScalarEngine>::Fr> {
    (0..count)
        .map(|_| <E as ScalarEngine>::Fr::random(rng))
        .collect()
}

fn dummy_vk<E: Engine, R: Rng>(public: usize, rng: &mut R) -> VerifyingKey<E> {
    VerifyingKey {
        alpha_g1: E::G1::random(rng).into_affine(),
        beta_g1: E::G1::random(rng).into_affine(),
        beta_g2: E::G2::random(rng).into_affine(),
        gamma_g2: E::G2::random(rng).into_affine(),
        delta_g1: E::G1::random(rng).into_affine(),
        delta_g2: E::G2::random(rng).into_affine(),
        ic: random_points::<E::G1, _>(public + 1, rng),
    }
}

fn dummy_params<E: Engine, R: Rng>(public: usize, private: usize, rng: &mut R) -> Parameters<E> {
    let count = public + private;
    let hlen = (1 << (((count + public + 1) as f64).log2().ceil() as usize)) - 1;
    Parameters {
        vk: dummy_vk(public, rng),
        h: Arc::new(random_points::<E::G1, _>(hlen, rng)),
        l: Arc::new(random_points::<E::G1, _>(private, rng)),
        a: Arc::new(random_points::<E::G1, _>(count, rng)),
        b_g1: Arc::new(random_points::<E::G1, _>(count, rng)),
        b_g2: Arc::new(random_points::<E::G2, _>(count, rng)),
    }
}

#[derive(Debug, StructOpt, Clone, Copy)]
#[structopt(name = "Bellman Bench", about = "Benchmarking Bellman.")]
struct Opts {
    #[structopt(long = "proofs", default_value = "1")]
    proofs: usize,
    #[structopt(long = "public", default_value = "1")]
    public: usize,
    #[structopt(long = "private", default_value = "1000000")]
    private: usize,
    #[structopt(long = "samples", default_value = "10")]
    samples: usize,
    #[structopt(long = "gpu")]
    gpu: bool,
    #[structopt(long = "verify")]
    verify: bool,
    #[structopt(long = "prove")]
    prove: bool,
    #[structopt(long = "dummy")]
    dummy: bool,
}

fn main() {
    let rng = &mut thread_rng();
    pretty_env_logger::init_timed();

    let opts = Opts::from_args();
    if opts.gpu {
        std::env::set_var("BELLMAN_VERIFIER", "gpu");
    } else {
        std::env::set_var("BELLMAN_NO_GPU", "1");
    }

    let circuit = DummyDemo {
        public: opts.public,
        private: opts.private,
    };
    let circuits = vec![circuit.clone(); opts.proofs];

    let params = if opts.dummy {
        dummy_params::<Bls12, _>(opts.public, opts.private, rng)
    } else {
        println!("Generating params... (You can skip this by passing `--dummy` flag)");
        generate_random_parameters(circuit.clone(), rng).unwrap()
    };
    let pvk = prepare_verifying_key(&params.vk);

    if opts.prove {
        println!("Proving...");

        for _ in 0..opts.samples {
            let (_, took) =
                timer!(create_random_proof_batch(circuits.clone(), &params, rng).unwrap());
            println!("Proof generation finished in {}ms", took);
        }
    }

    if opts.verify {
        println!("Verifying...");

        let (inputs, proofs) = if opts.dummy {
            (
                dummy_inputs::<Bls12, _>(opts.public, rng),
                dummy_proofs::<Bls12, _>(opts.proofs, rng),
            )
        } else {
            let mut inputs = Vec::new();
            let mut num = Fr::one();
            num.double();
            for _ in 0..opts.public {
                inputs.push(num);
                num.square();
            }
            println!("(Generating valid proofs...)");
            let proofs = create_random_proof_batch(circuits.clone(), &params, rng).unwrap();
            (inputs, proofs)
        };

        for _ in 0..opts.samples {
            let pref = proofs.iter().collect::<Vec<&_>>();
            println!(
                "{} proofs, each having {} public inputs...",
                opts.proofs, opts.public
            );
            let (valid, took) = timer!(verify_proofs_batch(
                &pvk,
                rng,
                &pref[..],
                &vec![inputs.clone(); opts.proofs]
            )
            .unwrap());
            println!("Verification finished in {}ms (Valid: {})", took, valid);
        }
    }
}
