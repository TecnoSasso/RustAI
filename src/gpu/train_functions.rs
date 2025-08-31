use crate::gpu::{*, structs::*};
use crate::common::enums::*;

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
        LossFunc::CrossEntropy => is_cross_entropy = true,
        _=> {
            is_cross_entropy = false;
            curr_act_on_cost = influences::loss_inf(&model.loss, &model.layers[model.layers.len()-1].a, &expected);
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
            curr_z_on_cost = influences::loss_inf(&LossFunc::CrossEntropy, &output, &expected);
        }
        else {
            curr_z_on_cost = influences::get_z_on_cost(
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