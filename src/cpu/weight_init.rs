use nalgebra::DMatrix;
use rand::{rng, Rng};
use rand_distr::Normal;

pub fn glorot(prev_size: usize, next_size: usize, nrows: usize, ncols: usize) -> DMatrix<f64>{
    let mut rng = rng();
    let scale = (6.0 / (prev_size + next_size) as f64).sqrt();
    DMatrix::<f64>::from_fn(nrows, ncols, |_, _| rng.random_range(-scale..=scale))
}
pub fn he(prev_size: usize, nrows: usize, ncols: usize) -> DMatrix<f64>{
    let mut rng = rng();
    let std_dev = (2.0 / prev_size as f64).sqrt();
    DMatrix::<f64>::from_fn(nrows, ncols, |_, _| rng.sample(Normal::new(0.0, std_dev).unwrap()))
}