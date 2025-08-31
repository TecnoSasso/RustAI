use nalgebra::{DVector, DMatrix};
use crate::common::enums::*;
use crate::cpu::enums::*;
use crate::cpu::structs::*;
use crate::cpu::*;

fn back_prop(model: &mut NN) -> (Vec<DVector<f64>>, Vec<DMatrix<f64>>){
    // First all the activation influences
    for i in (1..model.layers.len()).rev() {
        let (prev_slice, next_slice) = model.layers.split_at_mut(i);
        let prev_layer = &mut prev_slice[i-1];
        let curr_layer = &mut next_slice[0];

        let z_on_act: GradientType = influences::act_inf(&curr_layer.activ_func, &curr_layer.z);
        let act_on_cost: &DVector<f64> = &curr_layer.influences;
        for j in 0..=prev_layer.size-1 {
            let z_on_cost: DVector<f64>;

            match &z_on_act {
                GradientType::Vector(x) => {
                    z_on_cost = x.component_mul(act_on_cost);
                }
                GradientType::Matrix(x) => {
                    z_on_cost =  x * act_on_cost;
                }
            }
            
            prev_layer.influences[j] = curr_layer.weight_matrix.column(j).dot(&z_on_cost);
        }
    }

    // Ok, big part is done, now the bias gradient
    let mut biases_gradient: Vec<DVector<f64>> = [].to_vec();
    for i in (1..model.layers.len()).rev() {
        let curr_layer = &model.layers[i];
        let z_on_act: GradientType = influences::act_inf(&curr_layer.activ_func, &curr_layer.z);
        let act_on_cost: &DVector<f64> = &curr_layer.influences;
        let bias_grad: DVector<f64>;
        match &z_on_act {
            GradientType::Vector(x) => {
                bias_grad = x.component_mul(act_on_cost);
            }
            GradientType::Matrix(x) => {
                bias_grad =  x * act_on_cost;
            }
        }
        biases_gradient.insert(0, bias_grad);
    }

    // Then weight gradient (a bit tricky)
    let mut weight_gradient: Vec<DMatrix<f64>> = [].to_vec();
    for i in (1..=model.layers.len()-1).rev() {
        let (prev_slice, next_slice) = model.layers.split_at_mut(i);
        let prev_layer = &mut prev_slice[i-1];
        let curr_layer = &mut next_slice[0];

        let act_on_cost: &DVector<f64> = &curr_layer.influences;
        let z_on_act: GradientType = influences::act_inf(&curr_layer.activ_func, &curr_layer.z);

        let z_on_cost: DVector<f64>;  // const in a row
        match &z_on_act {
            GradientType::Vector(x) => {
                z_on_cost = act_on_cost.component_mul(&x);
            }
            GradientType::Matrix(x) => {
                z_on_cost =  x * act_on_cost;
            }
        }

        let weight_on_z: &DVector<f64> = &prev_layer.a;  // const in a column

        // Creating the matrix that stores the influences of the weights on the curr layer Zs
        let mut prev_on_current: DMatrix<f64> = DMatrix::from_columns(&vec![weight_on_z.clone(); curr_layer.size]);
        prev_on_current = prev_on_current.transpose();

        // Creating the matrix that stores the influences of the Zs on the final cost
        let curr_on_cost: DMatrix<f64> = DMatrix::from_columns(&vec![z_on_cost.clone(); prev_layer.size]);

        weight_gradient.insert(0, prev_on_current.component_mul(&curr_on_cost));
    }

    (biases_gradient, weight_gradient)
}

fn back_prop_train(model: &mut NN, batch: &Vec<(DVector<f64>, DVector<f64>)>, step: f64){
    let mut avrege_bias_gradient: Vec<DVector<f64>> = [].to_vec();
    let mut avrege_weight_gradient: Vec<DMatrix<f64>> = [].to_vec();
    for (input, expected) in batch{
        let ouput: DVector<f64> = model.ff(input, true);

        // Setting up activation influences for the last layer
        let model_len: usize = model.layers.len()-1;
        model.layers[model_len].influences = influences::loss_inf(&model.loss, &ouput, &expected);
        
        let (biases_gradien, weight_gradient) = back_prop(model);
        
        if avrege_bias_gradient.len() == 0{
            avrege_bias_gradient = biases_gradien;
        }
        else{
            for i in 0..biases_gradien.len(){
                avrege_bias_gradient[i] += &biases_gradien[i];
            }
        }

        if avrege_weight_gradient.len() == 0{
            avrege_weight_gradient = weight_gradient;
        }
        else{
            for i in 0..weight_gradient.len(){
                avrege_weight_gradient[i] += &weight_gradient[i];
            }
        }
    }
    avrege_bias_gradient = avrege_bias_gradient.iter().map(
        |x:&DVector<f64>| x.scale(1.0/batch.len() as f64)).collect();

    avrege_weight_gradient = avrege_weight_gradient.iter().map(
        |x:&DMatrix<f64>| x.scale(1.0/batch.len() as f64)).collect();

    // Update weight and biases
    for i in 1..model.layers.len(){
        model.layers[i].biases -= &avrege_bias_gradient[i-1].scale(step);
        model.layers[i].weight_matrix -= &avrege_weight_gradient[i-1].scale(step);
    }
}

pub fn train_model(model: &mut NN, batch: &Vec<(DVector<f64>, DVector<f64>)>, func: &TrainFunc, step: f64){
    match func {
        TrainFunc::BackProp => back_prop_train(model, batch, step),
        TrainFunc::NEAT => (/* TODO */),
    }
}