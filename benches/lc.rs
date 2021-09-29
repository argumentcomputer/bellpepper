use bellperson::{Index, LinearCombination, Variable};
use blstrs::Scalar as Fr;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ff::Field;

fn lc_benchmark(c: &mut Criterion) {
    c.bench_function("LinearCombination::add((Fr, Variable))", |b| {
        b.iter(|| {
            let mut lc = LinearCombination::<Fr>::zero();
            for i in 0..100 {
                let coeff = Fr::one();
                lc = lc + (coeff, Variable::new_unchecked(Index::Aux(i)));
            }
            black_box(lc);
        });
    })
    .bench_function("LinearCombination::add(LinearCombination)", |b| {
        let mut lc1 = LinearCombination::<Fr>::zero();
        let mut lc2 = LinearCombination::<Fr>::zero();
        for i in 0..10 {
            let coeff = Fr::one();
            lc1 = lc1 + (coeff, Variable::new_unchecked(Index::Aux(i)));

            let coeff = Fr::one();
            lc2 = lc2 + (coeff, Variable::new_unchecked(Index::Aux(i * 2)));
        }

        b.iter(|| {
            let mut lc = lc1.clone();
            for _ in 0..10 {
                lc = lc + &lc2;
            }
            black_box(lc);
        });
    });
}

criterion_group!(benches, lc_benchmark);
criterion_main!(benches);
