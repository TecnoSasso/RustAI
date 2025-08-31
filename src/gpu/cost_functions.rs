use cust::prelude::*;
use crate::gpu::{PTX, structs::*};
use crate::common::enums::*;

fn mean_squared_error(batch: &(Matrix, Matrix), model:&mut NN) -> f32{
    let (input, expected) = batch;

    let output: Matrix = model.ff(&input, false);
    
    if input.row_len != output.row_len {
        panic!("Input and output dimentions don't match");
    }
    if output.data.len() != expected.data.len() {
        panic!("Output and expected dimentions don't match");
    }

    let vec_c_d: DeviceBuffer<f32> = DeviceBuffer::zeroed(input.row_len).unwrap();
    
    // Kernel call
    let module = Module::from_ptx(PTX, &[]).unwrap();
    let function = module.get_function("Mean_Squared_Error").unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
    
    let nbloks_x: u32 = (output.row_len / 32 +1) as u32;
    
    unsafe {
        launch!(function<<<nbloks_x, 32 as u32, 0, stream>>>(
            vec_c_d.as_device_ptr(),
            output.data.as_device_ptr(),
            expected.data.as_device_ptr(),
            output.row_len as u32,
            output.data.len() as u32
        )).unwrap();
    }
    stream.synchronize().unwrap();
    
    let mut sum: f32 = 0.0;
    let mut vec_c: Vec<f32> = vec![0.0; vec_c_d.len()];
    vec_c_d.copy_to(&mut vec_c).unwrap();

    for i in 0..vec_c.len(){
        sum += vec_c[i];
    }
    sum / (vec_c.len() as f32)
}

fn cross_entropy(batch: &(Matrix, Matrix), model:&mut NN) -> f32{
    let (input, expected) = batch;

    let output: Matrix = model.ff(&input, false);

    if input.row_len != output.row_len {
        panic!("Input and output dimentions don't match");
    }
    
    let vec_c_d: DeviceBuffer<f32> = DeviceBuffer::zeroed(input.row_len).unwrap();
    
    // Kernel call
    let module = Module::from_ptx(PTX, &[]).unwrap();
    let function = module.get_function("Cross_Entropy").unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
    
    let nbloks_x: u32 = (output.row_len / 32 +1) as u32;
    
    unsafe {
        launch!(function<<<nbloks_x, 32 as u32, 0, stream>>>(
            vec_c_d.as_device_ptr(),
            output.data.as_device_ptr(),
            expected.data.as_device_ptr(),
            output.row_len as u32,
            output.data.len() as u32
        )).unwrap();
    }
    stream.synchronize().unwrap();
    
    let mut sum: f32 = 0.0;
    let mut vec_c: Vec<f32> = vec![0.0; vec_c_d.len()];
    vec_c_d.copy_to(&mut vec_c).unwrap();

    for i in &vec_c {
        sum += i;
    }
    sum / (vec_c.len() as f32)
}

pub fn eval_loss(func: &LossFunc, batch: &(Matrix, Matrix), model:&mut NN) -> f32{
    match func {
        LossFunc::MeanSquaredError => mean_squared_error(batch, model),
        LossFunc::CrossEntropy => cross_entropy(batch, model),
    }
}