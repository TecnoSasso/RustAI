use nalgebra::{DVector};
use crate::common::enums::*;

fn relu(x: &DVector<f64>) -> DVector<f64> {
    x.map(|v| if v > 0.0 { v } else { 0.0 })
}

fn sigmoid(x: &DVector<f64>) -> DVector<f64> {
    use std::f64::consts::E;
    x.map(|v| 1.0/(1.0+E.powf(-v)))
}

fn tanh(x: &DVector<f64>) -> DVector<f64> {
    use std::f64::consts::E;
    x.map(|v| {
        if v > 10.0 {1.0}
        else if v < -10.0 {-1.0}
        else {
            let a = E.powf(v);
            let b = E.powf(-v);
            (a - b) / (a + b)
        }
    })
}

fn softmax(x: &DVector<f64>) -> DVector<f64> {
    use std::f64::consts::E;
    let m = x.max();
    let tot = x.map(|a| E.powf(a-m)).sum();
    x.map(|v| {
        E.powf(v-m) / tot
    })
}

pub fn activate(act_fun: &Activation, x: &DVector<f64>) -> DVector<f64>{
    match act_fun {
        Activation::ReLu => return relu(x),
        Activation::Sigmoid => return sigmoid(x),
        Activation::Tanh => tanh(x),
        Activation::SoftMax => return softmax(x),
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::dvector;
    use super::*;

    #[test]
    fn test_relu() {
        assert_eq!(
            activate(&Activation::ReLu, &dvector![0.0, -2.0, -3.4, 3.4, 10_000.0]),
            dvector![0.0, 0.0, 0.0, 3.4, 10_000.0]);
    }

    #[test]
    fn test_sigmoid() {
        assert_eq!(
            activate(&Activation::Sigmoid, &dvector![0.0, -2.0, -3.4, 3.4, 10_000.0, -10_000.0]),
            dvector![0.5, 0.11920292202211757, 0.03229546469845052, 0.9677045353015494, 1.0, 0.0]);
    }

    #[test]
    fn test_tanh() {
        assert_eq!(
            activate(&Activation::Tanh, &dvector![0.0, -2.0, -3.4, 3.4, 10_000.0, -10_000.0, 10.0]),
            dvector![0.0, -0.964027580075817, -0.9977749279342794, 0.9977749279342794, 1.0, -1.0, 0.9999999958776926]);
    }

    #[test]
    fn test_softmax() {
        assert_eq!(
            activate(&Activation::SoftMax, &dvector![1.0, 2.0, 3.0]),
            dvector![0.09003057317038046, 0.24472847105479764, 0.6652409557748218]);
        assert_eq!(
            activate(&Activation::SoftMax, &dvector![10.0, -2.0, -3.4, 3.4, 1.1]),
            dvector![0.9984978435481867, 6.134982785100351e-6, 1.5128681286397673e-6, 0.0013583245519234511, 0.0001361840489761406]);
    }
}
