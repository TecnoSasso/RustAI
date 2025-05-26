use networks::NN;
use networks::Layer;
use networks::Activation::*;
use networks::LossFunc::*;
use networks::TrainFunc::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use nalgebra::{DVector};

fn one_hot_enc(n: usize, n_max: usize) -> DVector<f64> {
    let mut hot_vec: Vec<f64> = [].to_vec();
    for i in 0..n_max{
        if n == i{
            hot_vec.push(1.0);
        }
        else{
            hot_vec.push(0.0);
        }
    }
    DVector::from_vec(hot_vec)
}

fn main() {

    let mut model = NN {
        layers: vec![
            Layer::linear(784, Tanh),
            Layer::linear(16, Tanh),
            Layer::linear(16, Tanh),
            Layer::linear(10, SoftMax),
            ],
            loss: CrossEntropy
        };
        
        model.compile();
        
    // READ TRAINING DATA

    let mut train_data: Vec<(DVector<f64>, DVector<f64>)> = [].to_vec();
    let file = File::open("MNIST_DataSet/mnist_train.csv").unwrap();
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line.unwrap();
        train_data.push((DVector::from_vec(
            line.split_terminator(",").map(|x| x.parse::<f64>().unwrap() / 255.0).collect::<Vec<_>>().split_at(1).1.to_vec()
        ),
            one_hot_enc(line.split_terminator(",").collect::<Vec<_>>().to_vec().split_at(1).0[0].parse().unwrap(), 10)
        ));
    }

    // TRAIN MODEL

    model.train(&train_data, 200, BackProp, 1.0);
}