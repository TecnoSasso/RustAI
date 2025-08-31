use nalgebra::{DVector, DMatrix};
use crate::common::enums::*;
use crate::cpu::structs::*;
use crate::cpu::*;

impl Layer {
    pub fn linear(size: usize, activ_func: Activation) -> Self {
        Self {
            size,
            activ_func,
            weight_matrix: DMatrix::zeros(1, 1),
            biases: DVector::zeros(size),
            influences: DVector::zeros(size),
            z: DVector::zeros(size),
            a: DVector::zeros(size),
        }
    }

    pub fn init_matrix(&mut self, prev_size: usize, next_size: usize) {
        match self.activ_func{
            Activation::ReLu => self.weight_matrix = weight_init::he(prev_size, self.size, prev_size),
            
            _ => self.weight_matrix = weight_init::glorot(
                prev_size, next_size, self.size, prev_size),
        }
    }

    pub fn init_biases(&mut self){
        self.biases = DVector::<f64>::zeros(self.size);
    }
}