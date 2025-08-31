use cust::prelude::*;
use crate::gpu::{activations, structs::*};
use crate::common::enums::*;
use crate::gpu::PTX;

fn relu_inf(z: Matrix, act_on_cost: &Matrix) -> Matrix {

    let module = Module::from_ptx(PTX, &[]).unwrap();
    let function = module.get_function("ReLU_inf").unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    unsafe {
        launch!(function<<<(z.data.len() / 32 +1) as u32, 32 as u32, 0, stream>>>(
            z.data.as_device_ptr(),
            z.data.len(),
        )).unwrap();
    }

    stream.synchronize().unwrap();

    let z_on_act: Matrix = z;

    z_on_act.component_mul(act_on_cost)  // calculating z_on_cost
}

fn sigmoid_inf(z: Matrix, act_on_cost: &Matrix) -> Matrix {
    let module = Module::from_ptx(PTX, &[]).unwrap();
    let function = module.get_function("Sigmoid_inf").unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    unsafe {
        launch!(function<<<(z.data.len() / 32 +1) as u32, 32 as u32, 0, stream>>>(
            z.data.as_device_ptr(),
            z.data.len(),
        )).unwrap();
    }

    stream.synchronize().unwrap();

    let z_on_act: Matrix = z;

    z_on_act.component_mul(act_on_cost)  // calculating z_on_cost
}

fn tanh_inf(z: Matrix, act_on_cost: &Matrix) -> Matrix {
    let module = Module::from_ptx(PTX, &[]).unwrap();
    let function = module.get_function("Tanh_inf").unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    unsafe {
        launch!(function<<<(z.data.len() / 32 +1) as u32, 32 as u32, 0, stream>>>(
            z.data.as_device_ptr(),
            z.data.len(),
        )).unwrap();
    }

    stream.synchronize().unwrap();

    let z_on_act: Matrix = z;

    z_on_act.component_mul(act_on_cost)  // calculating z_on_cost
}

fn softmax_inf(z: Matrix, act_on_cost: &Matrix) -> Matrix {
    activations::activate(&Activation::SoftMax, &z);
    let dot_arr: &Matrix = &z.transpose().multiply(&act_on_cost).diagonal();
    (act_on_cost.transpose().add_bias(&dot_arr.scale(-1.0))).transpose().component_mul(&z)
}

pub fn get_z_on_cost(act_fun: &Activation, z: Matrix, act_on_cost: &Matrix) -> Matrix {

    match act_fun {
        Activation::ReLu => relu_inf(z, act_on_cost),
        Activation::Sigmoid => sigmoid_inf(z, act_on_cost),
        Activation::Tanh => tanh_inf(z, act_on_cost),
        Activation::SoftMax => softmax_inf(z, act_on_cost),
    }
}

// COST FUNCTION 

fn mse_inf(o: &Matrix, y: &Matrix) -> Matrix {
    o.component_add(&y.scale(-1.0)).scale(2.0)
}

fn cross_entropy_inf(p: &Matrix, y: &Matrix) -> Matrix {
    p.component_add(&y.scale(-1.0))
}

pub fn loss_inf(loss_fun: &LossFunc, o: &Matrix, y: &Matrix) -> Matrix {
    match loss_fun {
        LossFunc::MeanSquaredError => mse_inf(o, y),
        LossFunc::CrossEntropy => cross_entropy_inf(o, y),
    }
}