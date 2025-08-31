use rand::{rng, Rng};
use rand_distr::Normal;

use crate::gpu::structs::Matrix;

pub fn glorot(prev_size: usize, next_size: usize, nrows: usize, ncols: usize) -> Matrix {
    let mut rng = rng();
    let scale: f32 = (6.0 / (prev_size + next_size) as f32).sqrt();
    let weights: Vec<f32> = Vec::from_iter((0..nrows*ncols).map(|_| rng.random_range(-scale..=scale)));
    Matrix::from_vec(&weights, ncols)
}
pub fn he(prev_size: usize, nrows: usize, ncols: usize) -> Matrix {
    let mut rng = rng();
    let std_dev = (2.0 / prev_size as f32).sqrt();
    let distr: Normal<f32> = Normal::new(0.0, std_dev).unwrap();
    let weights: Vec<f32> = Vec::from_iter((0..nrows*ncols).map(|_| rng.sample(distr)));
    Matrix::from_vec(&weights, ncols)
}