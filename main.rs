use networks::NN;
use networks::Layer;
use networks::Activation::*;
use networks::CostFunc::*;
use networks::TrainFunc::*;
use nalgebra::DVector;

fn main() {
    let mut model = NN {
        layers: vec![
            Layer::linear(2, ReLu),
            Layer::linear(3, Sigmoid),
            Layer::linear(4, Sigmoid),
            Layer::linear(5, Sigmoid),
            Layer::linear(1, ReLu)
        ],
        cost: MeanSquaredError
    };
    
    model.compile();

    let batch: Vec<(DVector<f64>, DVector<f64>)> = [
        (DVector::from_vec(vec![0.0, 0.0]), DVector::from_vec(vec![0.0])),
        (DVector::from_vec(vec![1.0, 0.0]), DVector::from_vec(vec![1.0])),
        (DVector::from_vec(vec![0.0, 1.0]), DVector::from_vec(vec![1.0])),
        (DVector::from_vec(vec![1.0, 1.0]), DVector::from_vec(vec![0.0])),
    ].to_vec();

    model.train(batch, BackProp);
}