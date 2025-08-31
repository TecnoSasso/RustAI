use cust::{launch, memory::{CopyDestination, DeviceBuffer}, module::Module, stream::{Stream, StreamFlags}};
use crate::gpu::{*, structs::*};

pub fn categoral_accuracy(batch: &(Matrix, Matrix), model: &mut NN) -> f32 {
    
    let d_output: Matrix = model.ff(&batch.0, false);
    let d_expected: &Matrix = &batch.1;

    if d_output.row_len != d_expected.row_len {
        panic!("Input and expected output dimentions are not equal!")
    }
    
    let vec_c_d: DeviceBuffer<bool> = DeviceBuffer::zeroed(d_output.row_len).unwrap();
    let module = Module::from_ptx(PTX, &[]).unwrap();
    let function = module.get_function("Accuracy").unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
    
    let nbloks_x: u32 = (d_output.row_len / 32 +1) as u32;
    
    unsafe {
        launch!(function<<<nbloks_x, 32 as u32, 0, stream>>>(
            vec_c_d.as_device_ptr(),
            d_output.data.as_device_ptr(),
            d_expected.data.as_device_ptr(),
            d_output.row_len as u32,
            d_output.data.len() as u32
        )).unwrap();
    }
    
    stream.synchronize().unwrap();
    
    let mut output: Vec<bool> = vec![false; vec_c_d.len()];
    vec_c_d.copy_to(&mut output).unwrap();
    
    let mut sum: i32 = 0;
    for i in output{
        if i {
            sum += 1
        }
    }
    sum as f32 / d_expected.row_len as f32 * 100.0
}