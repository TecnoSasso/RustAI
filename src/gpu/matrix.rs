use std::backtrace::Backtrace;
use cust::{prelude::*};
use cust::memory::{DeviceBox, DeviceBuffer};
use cust::stream::{Stream, StreamFlags};
use crate::gpu::structs::*;
use crate::gpu::*;

impl Matrix {
    
    pub fn new (data: DeviceBuffer<f32>, row_len: usize) -> Self {
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

    pub fn from_vec (data: &Vec<f32>, row_len: usize) -> Self {
        Matrix::new(
            DeviceBuffer::from_slice(data.as_slice()).unwrap(),
            row_len
        )
    }

    pub fn zeroed(col_len: usize, row_len: usize) -> Self {
        let data: DeviceBuffer<f32> = DeviceBuffer::zeroed(col_len*row_len).unwrap();
        Self { data: data, row_len: row_len }
    }

    pub fn from_rows(rows: &Vec<&Matrix>) -> Self {
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

    pub fn as_device_box(&self) -> DeviceBox<DeviceMatrix> {
        let device_matrix = DeviceMatrix {
            data: self.data.as_device_ptr(),
            row_len: self.row_len as u32,
            data_len: self.data.len() as u32
        };
        DeviceBox::new(&device_matrix).unwrap()
    }

    pub fn clone(&self) -> Self {
        let mut dest: DeviceBuffer<f32> = DeviceBuffer::zeroed(self.data.len()).unwrap();
        dest.copy_from(&self.data);
        Self { data: dest, row_len: self.row_len }
    }

    pub fn scale(&self, scale: f32) -> Self {

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

    pub fn offset(&self, offset: f32) -> Self {

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

    pub fn dot(&self, vec_b: &Matrix) -> Self {

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

    pub fn component_add(&self, mat_b: &Matrix) -> Self {

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

    pub fn component_mul(&self, mat_b: &Matrix) -> Self {

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

    pub fn component_div(&self, mat_b: &Matrix) -> Self {

        if self.row_len != mat_b.row_len || self.data.len() != mat_b.data.len(){
            panic!("Matrix size unmatched");
        }

        // Memory allocation
        let d_mat_c: DeviceBuffer<f32> = DeviceBuffer::zeroed(self.data.len()).unwrap();

        let module = Module::from_ptx(PTX, &[]).unwrap();
        let function = module.get_function("component_div").unwrap();
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

    pub fn multiply(&self, mat_b: &Matrix) -> Self {
        
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

    pub fn transpose(&self) -> Self {
        let nbloks_x: u32 = (self.row_len / 32 +1) as u32;
        let nbloks_y: u32 = ((self.data.len() / self.row_len) / 32 +1) as u32;

        let d_mat_c: DeviceBuffer<f32> = DeviceBuffer::zeroed(self.data.len()).unwrap();

        let module = Module::from_ptx(PTX, &[]).unwrap();
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

    pub fn diagonal(&self) -> Self {

        let d_vec_c: DeviceBuffer<f32> = DeviceBuffer::zeroed(self.row_len).unwrap();

        let module = Module::from_ptx(PTX, &[]).unwrap();
        let function = module.get_function("diagonal").unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            launch!(function<<<(self.row_len / 32 +1) as u32, 32 as u32, 0, stream>>>(
                self.data.as_device_ptr(),
                d_vec_c.as_device_ptr(),
                self.row_len as u32,
            )).unwrap();
        }

        stream.synchronize().unwrap();
        Matrix::new(d_vec_c, 1)
    }

    pub fn internal_mean(&self) -> Self {
        let nbloks_x: u32 = (self.row_len / 32 +1) as u32;
        let nbloks_y: u32 = ((self.data.len() / self.row_len) / 32 +1) as u32;

        let d_vec_c: DeviceBuffer<f32> = DeviceBuffer::zeroed(self.data.len() / self.row_len).unwrap();

        let module = Module::from_ptx(PTX, &[]).unwrap();
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

    pub fn add_bias(&self, bias: &Matrix) -> Self{

        if self.data.len()/self.row_len != bias.data.len() || bias.row_len != 1{
            panic!("Bias not valid");
        }

        let d_mat_c: DeviceBuffer<f32> = DeviceBuffer::zeroed(self.data.len()).unwrap();

        let nbloks_x: u32 = (self.row_len / 32 +1) as u32;
        let nbloks_y: u32 = ((self.data.len() / self.row_len) / 32 +1) as u32;

        let module = Module::from_ptx(PTX, &[]).unwrap();
        let function = module.get_function("Add_Bias").unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            launch!(function<<<(nbloks_x, nbloks_y), (32 as u32, 32 as u32), 0, stream>>>(
                self.data.as_device_ptr(),
                d_mat_c.as_device_ptr(),
                bias.data.as_device_ptr(),
                self.row_len,
                self.data.len(),
            )).unwrap();
        }

        stream.synchronize().unwrap();
        Matrix::new(d_mat_c, self.row_len)
    }

    pub fn print(&self){
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

    pub fn print_dim(&self){
        println!("{} X {}", self.row_len, self.data.len() / self.row_len);
    }

}
