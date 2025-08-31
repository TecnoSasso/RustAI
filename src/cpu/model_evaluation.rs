use nalgebra::DVector;
use crate::cpu::structs::*;

pub fn accuracy(batch: &Vec<(DVector<f64>, DVector<f64>)>, model: &mut NN) -> f64 {
    let mut sum: i32 = 0;
    for (input, expected) in batch{
        let output: DVector<f64> = model.ff(input, false);
        let index_o = output.iter().position(|&r| r == output.max()).unwrap();
        let index_e = expected.iter().position(|&r| r == expected.max()).unwrap();
        if index_o == index_e {
            sum += 1
        }
    }
    sum as f64 / batch.len() as f64 * 100.0
}