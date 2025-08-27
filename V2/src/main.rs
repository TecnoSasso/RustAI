mod rust_ai;
use cust::{prelude::*};

use rust_ai::*;
use rust_ai::Activation::*;
use rust_ai::LossFunc::*;
use rust_ai::TrainFunc::*;

use std::fs::File;
use std::io::{BufRead, BufReader};

fn one_hot_enc(n: usize, n_max: usize) -> Vec<f32> {
    let mut hot_vec: Vec<f32> = [].to_vec();
    for i in 0..n_max{
        if n == i{
            hot_vec.push(1.0);
        }
        else{
            hot_vec.push(0.0);
        }
    }
    hot_vec
}

fn main() {

    cust::init(CudaFlags::empty()).unwrap();
    let device = Device::get_device(0).unwrap();
    let _context = Context::new(device);

    let mut model = NN {
        layers: vec![
            Layer::linear(784, Sigmoid),
            Layer::linear(16, Sigmoid),
            Layer::linear(16, Sigmoid),
            Layer::linear(10, Sigmoid),
            ],
            loss: MeanSquaredError
        };
        
        model.compile();
        
    // READ TRAINING DATA

    let mut train_data: Vec<(Vec<f32>, Vec<f32>)> = vec![];
    let file = File::open("MNIST_DataSet/mnist_train.csv").unwrap();
    let reader = BufReader::new(file);

    let mut i = 0;
    for line in reader.lines() {
        if i > 200*500 {break};
        let line = line.unwrap();
        train_data.push((
            line.split_terminator(",").map(|x| x.parse::<f32>().unwrap() / 255.0).collect::<Vec<_>>().split_at(1).1.to_vec()
        ,
            one_hot_enc(line.split_terminator(",").collect::<Vec<_>>().to_vec().split_at(1).0[0].parse().unwrap(), 10)
        ));
        i+=1;
    }

    model.train(&train_data, 100, BackProp, 0.1);
}