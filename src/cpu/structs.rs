use nalgebra::{DVector, DMatrix};
use crate::common::enums::*;

pub struct NN{
    pub layers: Vec<Layer>,
    pub loss: LossFunc,
}

pub struct Layer{
    pub size: usize,
    pub activ_func: Activation,
    pub weight_matrix: DMatrix<f64>,
    pub biases: DVector<f64>,
    pub influences: DVector<f64>,
    pub z: DVector<f64>,
    pub a: DVector<f64>,
}