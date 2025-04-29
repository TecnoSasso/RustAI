use nalgebra::{DMatrix, DVector};
use rand::Rng;

// MODULES --------------------------------

mod activations {
    use nalgebra::DVector;

    use crate::Activation;
    fn relu(x: &DVector<f64>) -> DVector<f64> {
        x.map(|v| if v > 0.0 { v } else { 0.0 })
    }
    
    fn sigmoid(x: &DVector<f64>) -> DVector<f64> {
        use std::f64::consts::E;
        x.map(|v| 1.0/(1.0+E.powf(-v)))
    }

    pub fn activate(act_fun: &Activation, x: &DVector<f64>) -> DVector<f64>{
        match act_fun {
            Activation::ReLu => return relu(x),
            Activation::Sigmoid => return sigmoid(x),
        }
    }
}

mod cost_functions {
    use crate::DVector;
    use crate::{CostFunc, NN};

    fn mean_squared_error(batch:Vec<(DVector<f64>, DVector<f64>)>, model:&mut NN) -> f64{
        let mut sum:f64 = 0.0;
        for (input, expected) in &batch{
            let ouput: DVector<f64> = model.ff(input, false);
            let mut cost_vec: DVector<f64> = ouput - expected;
            cost_vec = cost_vec.map(|x| x.powi(2));
            sum += cost_vec.sum() as f64;
        }
        sum / (batch.len() as f64)
    }

    pub fn eval_cost(func: CostFunc, batch:Vec<(DVector<f64>, DVector<f64>)>, model:&mut NN) -> f64{
        match func {
            CostFunc::MeanSquaredError => mean_squared_error(batch, model)
        }
    }
}

mod influences {
    use nalgebra::DVector;

    use crate::Activation;
    fn relu_inf(x: &DVector<f64>) -> DVector<f64> {
        x.map(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }
    
    fn sigmoid_inf(x: &DVector<f64>) -> DVector<f64> {
        use std::f64::consts::E;
        x.map(|v|{
            let sig = 1.0/(1.0+E.powf(-v));
            sig*(1.0-sig)
        })
    }

    pub fn act_inf(act_fun: &Activation, x: &DVector<f64>) -> DVector<f64>{
        match act_fun {
            Activation::ReLu => return relu_inf(x),
            Activation::Sigmoid => return sigmoid_inf(x),
        }
    }
}

mod train_functions {
    use nalgebra::DMatrix;

    use crate::influences::act_inf;
    use crate::TrainFunc;
    use crate::NN;
    use crate::DVector;
    
    fn back_prop(model: &mut NN) -> (Vec<DVector<f64>>, Vec<DMatrix<f64>>){
        // First all the activation influences
        for i in (1..=model.layers.len()-1).rev() {
            let (prev_slice, next_slice) = model.layers.split_at_mut(i);
            let prev_layer = &mut prev_slice[i-1];
            let curr_layer = &mut next_slice[0];

            let z_on_act: &DVector<f64> = &act_inf(&curr_layer.activ_func, &curr_layer.z);
            let act_on_cost: &DVector<f64> = &curr_layer.influences;
            for j in 0..=prev_layer.size-1 {
                let z_on_cost: DVector<f64> = z_on_act.component_mul(act_on_cost);
                prev_layer.influences[j] = curr_layer.weight_matrix.column(j).dot(&z_on_cost);
            }
        }

        // Ok, big part is done, now the bias gradient
        let mut biases_gradient: Vec<DVector<f64>> = [].to_vec();
        for i in (1..=model.layers.len()-1).rev() {
            let curr_layer = &model.layers[i];
            let z_on_act: &DVector<f64> = &act_inf(&curr_layer.activ_func, &curr_layer.z);
            let act_on_cost: &DVector<f64> = &curr_layer.influences;
            let bias_grad: DVector<f64> = z_on_act.component_mul(act_on_cost);
            biases_gradient.push(bias_grad);
        }

        // Then weight gradient (a bit tricky)
        let mut weight_gradient: Vec<DMatrix<f64>> = [].to_vec();
        for i in (1..=model.layers.len()-1).rev() {
            let (prev_slice, next_slice) = model.layers.split_at_mut(i);
            let prev_layer = &mut prev_slice[i-1];
            let curr_layer = &mut next_slice[0];

            let act_on_cost: &DVector<f64> = &curr_layer.influences;
            let z_on_act: &DVector<f64> = &act_inf(&curr_layer.activ_func, &curr_layer.z);

            let z_on_cost: &DVector<f64> = &act_on_cost.component_mul(z_on_act);  // const in a column
            let weight_on_z: &DVector<f64> = &prev_layer.a;  // const in a row

            // Creating the matrix that stores the influences of the weights on the curr layer Zs
            let mut prev_on_current: DMatrix<f64> = DMatrix::from_columns(&vec![weight_on_z.clone(); curr_layer.size]);
            prev_on_current = prev_on_current.transpose();

            // Creating the matrix that stores the influences of the Zs on the final cost
            let curr_on_cost: DMatrix<f64> = DMatrix::from_columns(&vec![z_on_cost.clone(); prev_layer.size]);

            weight_gradient.push(prev_on_current.component_mul(&curr_on_cost));
        }

        (biases_gradient, weight_gradient)
    }

    fn back_prop_train(model: &mut NN, batch: Vec<(DVector<f64>, DVector<f64>)>){
        for (input, expected) in &batch{
            let ouput: DVector<f64> = model.ff(input, true);
            let cost_vec: DVector<f64> = ouput - expected;
            // Setting up activation influences for the last layer
            let model_len: usize = model.layers.len()-1;
            model.layers[model_len].influences = cost_vec.map(|x| 2.0*x);
        }
    }

    pub fn train_model(model: &mut NN, batch: Vec<(DVector<f64>, DVector<f64>)>, func: TrainFunc){
        match func {
            TrainFunc::BackProp => back_prop_train(model, batch),
            TrainFunc::NEAT => (),
        }
    }
}

// ENUMS -----------------------------
pub enum Activation {
    ReLu,
    Sigmoid,
}

#[derive(Clone)]
pub enum CostFunc {
    MeanSquaredError,
}

pub enum TrainFunc {
    BackProp,
    NEAT,
}

// STRUCTS ---------------------------

pub struct NN{
    pub layers: Vec<Layer>,
    pub cost: CostFunc,
}

impl NN {
    pub fn ff(&mut self, input: &DVector<f64>, record: bool) -> DVector<f64>{
        let mut act_vec: DVector<f64> = input.clone();
        for i in 1..self.layers.len() {
            let cl = &mut self.layers[i];
            act_vec = &cl.weight_matrix * act_vec;
            act_vec += &cl.biases;
            if record { cl.z = act_vec.clone(); }  // for backprop
            act_vec = activations::activate(&cl.activ_func, &act_vec);
            if record { cl.a = act_vec.clone(); }  // for backprop
        }
        act_vec
    }

    pub fn cost(&mut self, batch:Vec<(DVector<f64>, DVector<f64>)>) -> f64{
        cost_functions::eval_cost(self.cost.clone(), batch, self)
    }

    pub fn compile(&mut self){
        // Create all the weight matrices
        if self.layers.len() < 1 {
            panic!("Not enough layers");
        }
        for i in 1..self.layers.len() {
            let prev_size = self.layers[i-1].size;
            self.layers[i].init_matrix(prev_size);
        }
        // Adding biases
        for i in 1..self.layers.len() {
            self.layers[i].init_biases();
        }
    }
    
    pub fn train(&mut self, batch: Vec<(DVector<f64>, DVector<f64>)>, func: TrainFunc){
        train_functions::train_model(self, batch, func);
    }

    pub fn print(&self){
        println!("-------------------------------------");
        for i in 1..self.layers.len() {
            println!("{}", self.layers[i].weight_matrix)
        }
        println!("-------------------------------------");
    }
}

pub struct Layer{
    pub size: usize,
    pub activ_func: Activation,
    weight_matrix: DMatrix<f64>,
    biases: DVector<f64>,
    influences: DVector<f64>,
    z: DVector<f64>,
    a: DVector<f64>,  // And there it goes the ram
}


impl Layer {
    pub fn linear(size: usize, activ_func: Activation) -> Self {
        Self {
            size,
            activ_func,
            weight_matrix: DMatrix::zeros(1, 1),
            biases: DVector::zeros(size),
            influences: DVector::zeros(size),
            z: DVector::zeros(size),
            a: DVector::zeros(size),
        }
    }

    fn init_matrix(&mut self, prev_size: usize) {
        let mut rng = rand::rng();
        let rows = self.size;
        let cols = prev_size;
        self.weight_matrix = DMatrix::<f64>::from_fn(rows, cols, |_, _| rng.random_range(0.0..=1.0));
    }

    fn init_biases(&mut self){
        let mut rng = rand::rng();
        self.biases = DVector::<f64>::from_fn(self.size, |_, _| rng.random_range(-1.0..=1.0));
    }
}