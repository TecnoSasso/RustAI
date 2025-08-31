use cust::prelude::*;
use crate::gpu::{PTX, structs::Matrix};
use crate::common::enums::*;

fn relu(x: &Matrix) {
    let module = Module::from_ptx(PTX, &[]).unwrap();
    let function = module.get_function("ReLU").unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    unsafe {
        launch!(function<<<(x.data.len() / 32 +1) as u32, 32 as u32, 0, stream>>>(
            x.data.as_device_ptr(),
            x.data.len()
        )).unwrap();
    }

    stream.synchronize().unwrap();
}

fn sigmoid(x: &Matrix) {
    let module = Module::from_ptx(PTX, &[]).unwrap();
    let function = module.get_function("Sigmoid").unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    unsafe {
        launch!(function<<<(x.data.len() / 32 +1) as u32, 32 as u32, 0, stream>>>(
            x.data.as_device_ptr(),
            x.data.len()
        )).unwrap();
    }

    stream.synchronize().unwrap();
}

fn tanh(x: &Matrix) {
    let module = Module::from_ptx(PTX, &[]).unwrap();
    let function = module.get_function("Tanh").unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    unsafe {
        launch!(function<<<(x.data.len() / 32 +1) as u32, 32 as u32, 0, stream>>>(
            x.data.as_device_ptr(),
            x.data.len()
        )).unwrap();
    }

    stream.synchronize().unwrap();
}

fn softmax(x: &Matrix) {
    let module = Module::from_ptx(PTX, &[]).unwrap();
    let function = module.get_function("SoftMax").unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    let nbloks_x: u32 = (x.row_len / 32 +1) as u32;

    unsafe {
        launch!(function<<<(nbloks_x, 1), (32 as u32, 1), 0, stream>>>(
            x.data.as_device_ptr(),
            x.row_len as u32,
            x.data.len() as u32
        )).unwrap();
    }

    stream.synchronize().unwrap();
}

pub fn activate(act_fun: &Activation, x: &Matrix) {
    match act_fun {
        Activation::ReLu => return relu(x),
        Activation::Sigmoid => return sigmoid(x),
        Activation::Tanh => tanh(x),
        Activation::SoftMax => return softmax(x),
    }
}