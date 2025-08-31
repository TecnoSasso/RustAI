use cust::memory::{DeviceBuffer, DevicePointer, DeviceCopy};
use crate::common::enums::*;
pub struct Matrix {
    pub data: DeviceBuffer<f32>,
    pub row_len: usize,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, DeviceCopy)]
pub struct DeviceMatrix {
    pub data: DevicePointer<f32>,
    pub row_len: u32,
    pub data_len: u32
}

pub struct NN{
    pub layers: Vec<Layer>,
    pub loss: LossFunc,
}

pub struct Layer{
    pub size: usize,
    pub activ_func: Activation,
    pub weight_matrix: Matrix,
    pub biases: Matrix,
    pub z: Matrix,
    pub a: Matrix,
}