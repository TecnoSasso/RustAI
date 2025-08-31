use nalgebra::{DVector};
use crate::common::enums::*;
use crate::cpu::structs::*;
use crate::cpu::*;

#[allow(dead_code)]
impl NN {
    pub fn ff(&mut self, input: &DVector<f64>, record: bool) -> DVector<f64>{
        let mut act_vec: DVector<f64> = input.clone();
        if record { self.layers[0].a = input.clone(); }
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

    #[doc = "Returns the average cost of the network estimated by the loss function"]
    pub fn cost(&mut self, data: &Vec<(DVector<f64>, DVector<f64>)>, times: usize, cost_func: LossFunc) -> f64{
        let batch: Vec<(DVector<f64>, DVector<f64>)> = data[0..times].to_vec();
        cost_functions::eval_loss(cost_func, &batch, self)
    }

    pub fn accuracy(&mut self, data: &Vec<(DVector<f64>, DVector<f64>)>, times: usize) -> f64 {
        let batch: Vec<(DVector<f64>, DVector<f64>)> = data[0..times].to_vec();
        model_evaluation::accuracy(&batch, self)
    }

    #[doc = "Initializes the network weights and biases, does not return anything."]
    pub fn compile(&mut self){
        // Create all the weight matrices
        if self.layers.len() < 1 {
            panic!("Not enough layers");
        }

        match self.loss {
            LossFunc::CrossEntropy => {
                match self.layers.last().unwrap().activ_func {
                    Activation::SoftMax => (),
                    _ => panic!("Cross Entropy function must take a probability distribution as input"),
                }
            },
            _ => (),
        }

        for i in 1..self.layers.len() {

            let prev_size = self.layers[i-1].size;
            
            if i == self.layers.len()-1{
                self.layers[i].init_matrix(prev_size, 0);
            }
            else{
                let next_size = self.layers[i+1].size;
                self.layers[i].init_matrix(prev_size, next_size);
            }
        }
        // Adding biases
        for i in 1..self.layers.len() {
            self.layers[i].init_biases();
        }
    }
    
    #[doc = "Trains the network with its `TrainFunc`, does not return anything."]
    pub fn train(&mut self, train_data: &Vec<(DVector<f64>, DVector<f64>)>, batch_size: usize, func: TrainFunc, step: f64){

        let mut train_data_clone = train_data.clone();
        
        let n_baches = train_data_clone.len() / batch_size;
        let mut i = 0;

        while train_data_clone.len() > 0 {
            let mut batch: Vec<(DVector<f64>, DVector<f64>)> = [].to_vec();
            for _ in 0..batch_size{
                if let Some(example ) = train_data_clone.pop(){
                    batch.push(example);
                }
                else{ break; }
            }
            train_functions::train_model(self, &batch, &func, step);
            i+=1;
            let a = self.loss.clone();
            println!("Batch {} of {} completed  Accuracy: {:6.2},  Cost: {:.5}",
            i, n_baches, self.accuracy(train_data, if train_data.len() > 500 {500} else {train_data.len()}),
            self.cost(train_data, if train_data.len() > 500 {500} else {train_data.len()}, a));
        }
    }

    pub fn print(&self){
        println!("--------------- WEIGHTS ---------------");
        for i in 1..self.layers.len() {
            println!("{}", self.layers[i].weight_matrix)
        }
        println!("---------------- BIASES ---------------");
        for i in 1..self.layers.len() {
            println!("{}", self.layers[i].biases)
        }
        println!("---------------------------------------");

    }
}