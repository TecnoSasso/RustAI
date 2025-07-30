use std::backtrace::Backtrace;
use std::fmt::Debug;
use cust::{prelude::*};
use cust::memory::{DeviceBox, DeviceBuffer, DeviceCopy};
use cust::stream::{Stream, StreamFlags};

const PTX: &'static str = include_str!("../target/matrix_operation.ptx");

// MATRIX STUFF ----------------------

pub struct Matrix {
    data: DeviceBuffer<f32>,
    row_len: usize,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, DeviceCopy)]
struct DeviceMatrix {
    data: DevicePointer<f32>,
    row_len: u32,
    data_len: u32
}

impl Matrix {
    
    fn new (data: DeviceBuffer<f32>, row_len: usize) -> Self {
        if data.len() == 0 && row_len == 0 {
            let bt = Backtrace::capture();
            println!("ZERO MATRIX HAS BEEN CREATED FROM {:?}\n", bt);
            Self { data: data, row_len: row_len }
        }
        else if data.len() > 0 && row_len == 0{
            panic!("Matrix has invalid dimentions");
        }
        else if data.len() % row_len != 0 {
            panic!("Matrix is not rectangular");
        }
        else{
            Self { data: data, row_len: row_len }
        }
    }

    fn from_vec (data: &Vec<f32>, row_len: usize) -> Self {
        Matrix::new(
            DeviceBuffer::from_slice(data.as_slice()).unwrap(),
            row_len
        )
    }

    fn zeroed(col_len: usize, row_len: usize) -> Self {
        let data: DeviceBuffer<f32> = DeviceBuffer::zeroed(col_len*row_len).unwrap();
        Self { data: data, row_len: row_len }
    }

    fn from_rows(rows: &Vec<&Matrix>) -> Self {
        let mut matrix_vec: Vec<f32> = vec![];
        for row in rows{
            if row.data.len() != row.row_len {
                panic!("Coloumn is not a vector")
            }

            let mut h_row = vec![0.0; row.data.len()];
            row.data.copy_to(&mut h_row).unwrap();
            matrix_vec.extend(h_row);
        }
        Matrix::from_vec(&matrix_vec, rows.len())
    }

    fn as_device_box(&self) -> DeviceBox<DeviceMatrix> {
        let device_matrix = DeviceMatrix {
            data: self.data.as_device_ptr(),
            row_len: self.row_len as u32,
            data_len: self.data.len() as u32
        };
        DeviceBox::new(&device_matrix).unwrap()
    }

    fn clone(&self) -> Self {
        let mut dest: DeviceBuffer<f32> = DeviceBuffer::zeroed(self.data.len()).unwrap();
        dest.copy_from(&self.data);
        Self { data: dest, row_len: self.row_len }
    }

    fn scale(&self, scale: f32) -> Self {

        let mat_c: DeviceBuffer<f32> = DeviceBuffer::zeroed(self.data.len()).unwrap();

        let module = Module::from_ptx(PTX, &[]).unwrap();
        let function = module.get_function("scale").unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            let _ = launch!(function<<<self.data.len() as u32, 32 as u32, 0, stream>>>(
                self.data.as_device_ptr(),
                mat_c.as_device_ptr(),
                scale,
                self.data.len()
            ));
        }

        stream.synchronize().unwrap();
        Matrix::new(mat_c, self.row_len)
    }

    fn offset(&self, offset: f32) -> Self {

        let mat_c: DeviceBuffer<f32> = DeviceBuffer::zeroed(self.data.len()).unwrap();

        let module = Module::from_ptx(PTX, &[]).unwrap();
        let function = module.get_function("offset").unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            let _ = launch!(function<<<self.data.len() as u32, 32 as u32, 0, stream>>>(
                self.data.as_device_ptr(),
                mat_c.as_device_ptr(),
                offset,
                self.data.len()
            ));
        }

        stream.synchronize().unwrap();
        Matrix::new(mat_c, self.row_len)
    }

    fn dot(&self, vec_b: &Matrix) -> Self {

        if self.row_len != vec_b.row_len || self.data.len() != vec_b.data.len(){
            panic!("Matrix size unmatched");
        }

        // Memory allocation
        let d_dot: DeviceBuffer<f32> = DeviceBuffer::zeroed(self.row_len).unwrap();

        let module = Module::from_ptx(PTX, &[]).unwrap();
        let function = module.get_function("dot").unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            launch!(function<<<1 as u32, 1 as u32, 0, stream>>>(
                d_dot.as_device_ptr(),
                self.data.as_device_ptr(),
                vec_b.data.as_device_ptr(),
                self.row_len,
                self.data.len()
            )).unwrap();
        }

        stream.synchronize().unwrap();
        Matrix::new(d_dot, self.row_len)
        
    }

    fn component_add(&self, mat_b: &Matrix) -> Self {

        if self.row_len != mat_b.row_len || self.data.len() != mat_b.data.len(){
            panic!("Matrix size unmatched");
        }

        // Memory allocation
        let d_mat_c: DeviceBuffer<f32> = DeviceBuffer::zeroed(self.data.len()).unwrap();

        let module = Module::from_ptx(PTX, &[]).unwrap();
        let function = module.get_function("component_add").unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            launch!(function<<<self.data.len() as u32, 32 as u32, 0, stream>>>(
                self.data.as_device_ptr(),
                mat_b.data.as_device_ptr(),
                d_mat_c.as_device_ptr(),
                self.data.len()
            )).unwrap();
        }

        stream.synchronize().unwrap();

        Matrix::new(d_mat_c, self.row_len)
    }

    fn component_mul(&self, mat_b: &Matrix) -> Self {

        if self.row_len != mat_b.row_len || self.data.len() != mat_b.data.len(){
            panic!("Matrix size unmatched");
        }

        // Memory allocation
        let d_mat_c: DeviceBuffer<f32> = DeviceBuffer::zeroed(self.data.len()).unwrap();

        let module = Module::from_ptx(PTX, &[]).unwrap();
        let function = module.get_function("component_mul").unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            launch!(function<<<self.data.len() as u32, 32 as u32, 0, stream>>>(
                self.data.as_device_ptr(),
                mat_b.data.as_device_ptr(),
                d_mat_c.as_device_ptr(),
                self.data.len()
            )).unwrap();
        }

        stream.synchronize().unwrap();

        Matrix::new(d_mat_c, self.row_len)
    }

    fn multiply(&self, mat_b: &Matrix) -> Self {
        
        if self.row_len <= 0 || mat_b.row_len <= 0 {
            panic!("Matrix sizes not valid: {}, {}", self.row_len, mat_b.row_len);
        }
        if self.row_len != mat_b.data.len() / mat_b.row_len {
            panic!("Matrix size unmatched");
        }

        // Memory allocation
        let result_len: usize = (self.data.len() / self.row_len) * mat_b.row_len;
        let d_mat_c: DeviceBuffer<f32> = DeviceBuffer::zeroed(result_len).unwrap();

        let nbloks_x: u32 = (mat_b.row_len / 32 +1) as u32;
        let nbloks_y: u32 = ((self.data.len() / self.row_len) / 32 +1) as u32;

        let module = Module::from_ptx(PTX, &[]).unwrap();
        let function = module.get_function("multiply").unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            launch!(function<<<(nbloks_x, nbloks_y), (32 as u32, 32 as u32), 0, stream>>>(
                self.data.as_device_ptr(),
                mat_b.data.as_device_ptr(),
                d_mat_c.as_device_ptr(),
                self.row_len as u32,
                mat_b.row_len as u32,
                (self.data.len() / self.row_len) as u32
            )).unwrap();
        }

        stream.synchronize().unwrap();

        let mut debug_c: Vec<f32> = vec![0.0; d_mat_c.len()];
        d_mat_c.copy_to(&mut debug_c).unwrap();
        for i in debug_c {
            if i.abs() > 100.0 {
                panic!("CALCOLO ERRATO");
            }
        }

        Matrix::new(d_mat_c, mat_b.row_len)
    }

    fn transpose(&self) -> Self {
        let nbloks_x: u32 = (self.row_len / 32 +1) as u32;
        let nbloks_y: u32 = ((self.data.len() / self.row_len) / 32 +1) as u32;

        let d_mat_c: DeviceBuffer<f32> = DeviceBuffer::zeroed(self.data.len()).unwrap();

        let ptx = include_str!("../target/matrix_operation.ptx");
        let module = Module::from_ptx(ptx, &[]).unwrap();
        let function = module.get_function("transpose").unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            launch!(function<<<(nbloks_x, nbloks_y), (32 as u32, 32 as u32), 0, stream>>>(
                self.data.as_device_ptr(),
                d_mat_c.as_device_ptr(),
                self.row_len as u32,
                self.data.len(),
            )).unwrap();
        }

        stream.synchronize().unwrap();
        Matrix::new(d_mat_c, self.data.len() / self.row_len)
    }

    fn internal_mean(&self) -> Self {
        let nbloks_x: u32 = (self.row_len / 32 +1) as u32;
        let nbloks_y: u32 = ((self.data.len() / self.row_len) / 32 +1) as u32;

        let d_vec_c: DeviceBuffer<f32> = DeviceBuffer::zeroed(self.data.len() / self.row_len).unwrap();

        let ptx = include_str!("../target/matrix_operation.ptx");
        let module = Module::from_ptx(ptx, &[]).unwrap();
        let function = module.get_function("internal_mean").unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            launch!(function<<<(nbloks_x, nbloks_y), (32 as u32, 32 as u32), 0, stream>>>(
                self.data.as_device_ptr(),
                d_vec_c.as_device_ptr(),
                self.row_len as u32,
                self.data.len(),
            )).unwrap();
        }

        stream.synchronize().unwrap();
        Matrix::new(d_vec_c, 1)
    }

    fn add_bias(&self, bias: &Matrix){

        if self.data.len()/self.row_len != bias.data.len() || bias.row_len != 1{
            panic!("Bias not valid");
        }

        let nbloks_x: u32 = (self.row_len / 32 +1) as u32;
        let nbloks_y: u32 = ((self.data.len() / self.row_len) / 32 +1) as u32;

        let ptx = include_str!("../target/matrix_operation.ptx");
        let module = Module::from_ptx(ptx, &[]).unwrap();
        let function = module.get_function("Add_Bias").unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            launch!(function<<<(nbloks_x, nbloks_y), (32 as u32, 32 as u32), 0, stream>>>(
                self.data.as_device_ptr(),
                bias.data.as_device_ptr(),
                self.row_len,
                self.data.len(),
            )).unwrap();
        }

        stream.synchronize().unwrap();
    }

    fn print(&self){
        let mut mat: Vec<f32> = vec![0.0f32; self.data.len()];

        self.data.copy_to(&mut mat).unwrap();

        println!();
        for i in 0..mat.len() / self.row_len{
            print!("|  ");
            for j in 0..self.row_len {
                print!("{:^8.4}", mat[j + i*self.row_len]);
            }
            println!("|");
        }
        println!("");
    }

    fn print_dim(&self){
        println!("{} X {}", self.row_len, self.data.len() / self.row_len);
    }

}

// MODULES ---------------------------

mod weight_init {
    use rand::{rng, Rng};
    use rand_distr::Normal;

    use crate::rust_ai::Matrix;

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
}

mod activations {
    use super::PTX;

    use cust::prelude::*;
    use crate::{rust_ai::Matrix, Activation};

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

}

mod cost_functions {
    use core::f32;
    use std::vec;
    use super::Matrix;
    use cust::prelude::*;
    use super::PTX;

    use crate::{LossFunc, NN};

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
}

mod influences {
    use crate::{rust_ai::{activations::activate, Matrix}, Activation, LossFunc};
    use cust::prelude::*;

    fn relu_inf(z: Matrix, act_on_cost: &Matrix) -> Matrix {

        let ptx = include_str!("../target/matrix_operation.ptx");
        let module = Module::from_ptx(ptx, &[]).unwrap();
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

        let ptx = include_str!("../target/matrix_operation.ptx");
        let module = Module::from_ptx(ptx, &[]).unwrap();
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

        let ptx = include_str!("../target/matrix_operation.ptx");
        let module = Module::from_ptx(ptx, &[]).unwrap();
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
        activate(&Activation::SoftMax, &z);
        act_on_cost.component_add(&z.transpose().multiply(act_on_cost).scale(-1.0)).component_mul(&z)
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
}

mod train_functions {
    use core::panic;

    use super::influences::get_z_on_cost;
    use super::influences::loss_inf;
    use crate::rust_ai::Matrix;
    use crate::TrainFunc;
    use crate::NN;
    
    fn back_prop(model: &mut NN, batch: &(Matrix, Matrix), step: f32) {
        // Feed forward
        let input = &batch.0;
        let expected = &batch.1;
        model.ff(&input, true);

        // Actual backprop
        // Setting up activation influences for the last layer
        let is_cross_entropy: bool;
        let mut curr_act_on_cost: Matrix = Matrix::zeroed(0, 0);

        match model.loss {
            crate::LossFunc::CrossEntropy => is_cross_entropy = true,
            _=> {
                is_cross_entropy = false;
                curr_act_on_cost = loss_inf(&model.loss, &model.layers[model.layers.len()-1].a, &expected);
            },
        }

        // First all the activation influences
        let n_layers = model.layers.len();
        
        let output = &model.layers[model.layers.len()-1].a.clone();
        
        for i in (1..n_layers).rev() {
            let (prev_slice, next_slice) = model.layers.split_at_mut(i);
            let prev_layer = &mut prev_slice[i-1];
            let curr_layer = &mut next_slice[0];

            if curr_layer.a.row_len != prev_layer.a.row_len {
                panic!("Batch size not consistent");
            }
            
            let curr_z_on_cost: Matrix;
            if is_cross_entropy && i == n_layers-1 {
                curr_z_on_cost = loss_inf(&model.loss, &output, &expected);
            }
            else {
                curr_z_on_cost = get_z_on_cost(
                    &curr_layer.activ_func,
                    curr_layer.z.clone(),
                    &curr_act_on_cost
                );
            }

            // CALCULATING INFLUENCES OF ACTIVATIONS ON COST

            let prev_act_on_cost = curr_layer.weight_matrix.transpose().multiply(&curr_z_on_cost);

            // UPDATING BIASES
            let bias_grad = curr_z_on_cost.internal_mean().scale(-step);
            
            curr_layer.biases = curr_layer.biases.component_add(&bias_grad);
            
            // CALCUlATING INFLUENCES OF WEIGHTS ON COST

            let mut weight_on_cost = prev_layer.a.multiply(&curr_z_on_cost.transpose());

            weight_on_cost = weight_on_cost.transpose().scale(1.0/curr_layer.a.row_len as f32);
            
            // UPDATING WEIGHTS
            let weight_grad = weight_on_cost.scale(-step);

            curr_layer.weight_matrix = curr_layer.weight_matrix.component_add(&weight_grad);

            curr_act_on_cost = prev_act_on_cost
        }
    }

    pub fn train_model(model: &mut NN, batch: &(Matrix, Matrix), func: &TrainFunc, step: f32){
        match func {
            TrainFunc::BackProp => back_prop(model, batch, step),
            TrainFunc::NEAT => (/* TODO */),
        }
    }
}

mod model_evaluations {
    use cust::{launch, memory::{CopyDestination, DeviceBuffer}, module::Module, stream::{Stream, StreamFlags}};
    use crate::{rust_ai::PTX, NN};
    use super::Matrix;

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
}

// ENUMS -----------------------------
pub enum Activation {
    ReLu,
    Sigmoid,
    Tanh,
    SoftMax,
}

pub enum LossFunc {
    MeanSquaredError,
    CrossEntropy,
}

pub enum TrainFunc {
    BackProp,
    NEAT,
}

// STRUCTS ---------------------------

pub struct NN{
    pub layers: Vec<Layer>,
    pub loss: LossFunc,
}

impl NN {

    fn batch_to_matrix(batch_vector: &[(Vec<f32>, Vec<f32>)]) -> (Matrix, Matrix){
        if batch_vector.len() == 0 {
            panic!("Empty batch")
        }
        let examples_vector: Vec<Vec<f32>> = batch_vector.iter().map(|x| x.0.clone()).collect();
        let lables_vector: Vec<Vec<f32>> = batch_vector.iter().map(|x| x.1.clone()).collect();

        let flat_examples: Vec<f32> = examples_vector.into_iter().flat_map(|inner| inner).collect();
        let examples_matrix: Matrix = Matrix::from_vec(&flat_examples, batch_vector.len());

        let flat_labels: Vec<f32> = lables_vector.into_iter().flat_map(|inner| inner).collect();
        let labels_matrix: Matrix = Matrix::from_vec(&flat_labels, batch_vector.len());

        (examples_matrix, labels_matrix)
    }

    fn ff(&mut self, input: &Matrix, record: bool) -> Matrix {
        let mut act_mat: Matrix = input.clone();
        if record { self.layers[0].a = input.clone(); }
        for i in 1..self.layers.len() {
            let curr_layer = &mut self.layers[i];
            act_mat = curr_layer.weight_matrix.multiply(&act_mat);
            act_mat.add_bias(&curr_layer.biases);
            if record { curr_layer.z = act_mat.clone(); }  // for backprop
            activations::activate(&curr_layer.activ_func, &act_mat);
            if record { curr_layer.a = act_mat.clone(); }  // for backprop
        }
        println!("FEED FORWAD COMPLETED\n");
        act_mat
    }

    #[doc = "Returns the average cost of the network estimated by the loss function"]
    pub fn cost(&mut self, data: &Vec<(Vec<f32>, Vec<f32>)>, times: usize, cost_func: &LossFunc) -> f32{
        let rng = rand::rng();
        let batch: (Matrix, Matrix) = NN::batch_to_matrix(&data[0..times]);
        cost_functions::eval_loss(cost_func, &batch, self)
    }

    pub fn categoral_accuracy(&mut self, data: &Vec<(Vec<f32>, Vec<f32>)>, times: usize) -> f32 {
        let batch: (Matrix, Matrix) = NN::batch_to_matrix(&data[0..times]);
        model_evaluations::categoral_accuracy(&batch, self)
    }

    #[doc = "Initializes the network weights and biases, does not return anything."]
    pub fn compile(&mut self){
        // Create all the weight matrices
        if self.layers.len() < 1 {
            panic!("Not enough layers");
        }

        match self.loss {
            LossFunc::CrossEntropy => {
                match self.layers.last().unwrap().activ_func {
                    Activation::SoftMax => (),
                    _ => panic!("Cross Entropy function must take a probability distribution as input"),
                }
            },
            _ => (),
        }

        for i in 1..self.layers.len() {

            let prev_size = self.layers[i-1].size;
            
            if i == self.layers.len()-1{
                self.layers[i].init_matrix(prev_size, 0);
            }
            else{
                let next_size = self.layers[i+1].size;
                self.layers[i].init_matrix(prev_size, next_size);
            }
        }
    }
    
    #[doc = "Trains the network with its `TrainFunc`, does not return anything."]
    pub fn train(&mut self, train_data: &Vec<(Vec<f32>, Vec<f32>)>, batch_size: usize, func: TrainFunc, step: f32){
        let n_baches = train_data.len() / batch_size;

        for i in 0..train_data.len() / batch_size{
            let batch: (Matrix, Matrix) = NN::batch_to_matrix(&train_data[i*batch_size..i*batch_size+batch_size]);

            train_functions::train_model(self, &batch, &func, step);
            println!("Batch {} of {} completed  Cost: {:.2}", i+1, n_baches, self.cost(train_data, 50, &LossFunc::MeanSquaredError));
        }
    }

    pub fn print(&self){
        println!("--------------- WEIGHTS ---------------");
        for i in 1..self.layers.len() {
            self.layers[i].weight_matrix.print_dim();
        }
        println!("---------------- BIASES ---------------");
        for i in 1..self.layers.len() {
            self.layers[i].biases.print_dim();
        }
        println!("---------------------------------------");

    }
}

pub struct Layer{
    pub size: usize,
    pub activ_func: Activation,
    weight_matrix: Matrix,
    biases: Matrix,
    z: Matrix,
    a: Matrix,
}

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

    fn init_matrix(&mut self, prev_size: usize, next_size: usize) {
        match self.activ_func{
            Activation::ReLu => self.weight_matrix = weight_init::he(prev_size, self.size, prev_size),
            
            _ => self.weight_matrix = weight_init::glorot(
                prev_size, next_size, self.size, prev_size),
        }
    }
}