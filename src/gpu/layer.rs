use crate::gpu::{weight_init, structs::*};
use crate::common::enums::*;

impl Layer {
    pub fn linear(size: usize, activ_func: Activation) -> Self {
        Self {
            size,
            activ_func,
            weight_matrix: Matrix::zeroed(1, 1),
            biases: Matrix::zeroed(size, 1),
            z: Matrix::zeroed(0, 0),  // ff will take care of that
            a: Matrix::zeroed(0, 0),  // ff will take care of that
        }
    }

    pub fn init_matrix(&mut self, prev_size: usize, next_size: usize) {
        match self.activ_func{
            Activation::ReLu => self.weight_matrix = weight_init::he(prev_size, self.size, prev_size),
            
            _ => self.weight_matrix = weight_init::glorot(
                prev_size, next_size, self.size, prev_size),
        }
    }
}