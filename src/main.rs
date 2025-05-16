use networks::NN;
use networks::Layer;
use networks::Activation::*;
use networks::CostFunc::*;
use networks::TrainFunc::*;
use nalgebra::DVector;



fn main() {
    let mut model = NN {
        layers: vec![
            Layer::linear(2, Tanh),
            Layer::linear(4, Tanh),
            Layer::linear(4, Tanh),
            Layer::linear(2, SoftMax),
        ],
        cost: MeanSquaredError
    };
    
    model.compile();

    let train_data: Vec<(DVector<f64>, DVector<f64>)> = [
        (DVector::from_vec(vec![0.0, 0.0]), DVector::from_vec(vec![0.0, 1.0])),
        (DVector::from_vec(vec![1.0, 0.0]), DVector::from_vec(vec![1.0, 0.0])),
        (DVector::from_vec(vec![0.0, 1.0]), DVector::from_vec(vec![1.0, 0.0])),
        (DVector::from_vec(vec![1.0, 1.0]), DVector::from_vec(vec![0.0, 1.0])),
    ].to_vec();

    let mut cost: f64;
    for _ in 0..1000{
        cost = model.cost(&train_data);
        println!("{}", cost);
        model.train(&train_data, 4, BackProp, 1.0);
    }
}