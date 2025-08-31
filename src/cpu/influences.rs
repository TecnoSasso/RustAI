use nalgebra::{DMatrix, DVector};
use crate::common::enums::*;
use crate::cpu::enums::*;
use crate::cpu::*;

fn relu_inf(x: &DVector<f64>) -> GradientType {
    GradientType::Vector(x.map(|v| if v > 0.0 { 1.0 } else { 0.0 }))
}

fn sigmoid_inf(x: &DVector<f64>) -> GradientType {
    use std::f64::consts::E;
    GradientType::Vector(x.map(|v|{
        let sig = 1.0/(1.0+E.powf(-v));
        sig*(1.0-sig)
    }))
}

fn tanh_inf(x: &DVector<f64>) -> GradientType {
    use std::f64::consts::E;
    GradientType::Vector(x.map(|v| {
        let a = E.powf(v);
        let b = E.powf(-v);
        let t = (a - b) / (a + b);
        1.0 - t.powi(2)
    }))
}

pub fn softmax_inf(v: &DVector<f64>) -> GradientType {
    let a: DVector<f64> = activations::activate(&Activation::SoftMax, v);
    let mut cols: Vec<DVector<f64>> = [].to_vec();
    for i in 0..a.len() {
        cols.push(DVector::zeros(a.len()));
        for j in 0..a.len(){
            if i == j{
                cols[i][j] = a[i]*(1.0-a[i])
            }
            else{
                cols[i][j] = -a[i]*a[j]
            }
        }
    }
    GradientType::Matrix(DMatrix::from_columns(&cols))
}

pub fn act_inf(act_fun: &Activation, x: &DVector<f64>) -> GradientType{
    match act_fun {
        Activation::ReLu => relu_inf(x),
        Activation::Sigmoid => sigmoid_inf(x),
        Activation::Tanh => tanh_inf(x),
        Activation::SoftMax => softmax_inf(x),
    }
}

// COST FUNCTION 

fn mse_inf(o: &DVector<f64>, y: &DVector<f64>) -> DVector<f64>{
    (o-y).scale(2.0)
}

fn xen_inf(p: &DVector<f64>, y: &DVector<f64>) -> DVector<f64>{
    -y.component_div(p)
}

pub fn loss_inf(loss_fun: &LossFunc, o: &DVector<f64>, y: &DVector<f64>) -> DVector<f64>{
    match loss_fun {
        LossFunc::MeanSquaredError => mse_inf(o, y),
        LossFunc::CrossEntropy => xen_inf(o, y),
    }
}