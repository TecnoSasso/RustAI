use crate::gpu::{*, structs::*};
use crate::common::enums::*;

impl NN {

    pub fn batch_to_matrix(batch_vector: &[(Vec<f32>, Vec<f32>)]) -> (Matrix, Matrix){
        if batch_vector.len() == 0 {
            panic!("Empty batch")
        }
        let examples_vector: Vec<Vec<f32>> = batch_vector.iter().map(|x| x.0.clone()).collect();
        let lables_vector: Vec<Vec<f32>> = batch_vector.iter().map(|x| x.1.clone()).collect();

        let flat_examples: Vec<f32> = examples_vector.into_iter().flat_map(|inner| inner).collect();
        let examples_matrix: Matrix = Matrix::from_vec(&flat_examples, batch_vector.len());

        let flat_labels: Vec<f32> = lables_vector.into_iter().flat_map(|inner| inner).collect();
        let labels_matrix: Matrix = Matrix::from_vec(&flat_labels, batch_vector.len());

        (examples_matrix, labels_matrix)
    }

    pub fn ff(&mut self, input: &Matrix, record: bool) -> Matrix {
        let mut act_mat: Matrix = input.clone();
        if record { self.layers[0].a = input.clone(); }
        for i in 1..self.layers.len() {
            let curr_layer = &mut self.layers[i];
            act_mat = curr_layer.weight_matrix.multiply(&act_mat);
            act_mat = act_mat.add_bias(&curr_layer.biases);
            if record { curr_layer.z = act_mat.clone(); }  // for backprop
            activations::activate(&curr_layer.activ_func, &act_mat);
            if record { curr_layer.a = act_mat.clone(); }  // for backprop
        }
        act_mat
    }

    #[doc = "Returns the average cost of the network estimated by the loss function"]
    pub fn cost(&mut self, data: &Vec<(Vec<f32>, Vec<f32>)>, times: usize, cost_func: &LossFunc) -> f32{
        let batch: (Matrix, Matrix) = NN::batch_to_matrix(&data[0..times]);
        cost_functions::eval_loss(cost_func, &batch, self)
    }

    pub fn categoral_accuracy(&mut self, data: &Vec<(Vec<f32>, Vec<f32>)>, times: usize) -> f32 {
        let batch: (Matrix, Matrix) = NN::batch_to_matrix(&data[0..times]);
        model_evaluation::categoral_accuracy(&batch, self)
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
    }
    
    #[doc = "Trains the network with its `TrainFunc`, does not return anything."]
    pub fn train(&mut self, train_data: &Vec<(Vec<f32>, Vec<f32>)>, batch_size: usize, func: TrainFunc, step: f32){
        let n_baches = train_data.len() / batch_size;

        for i in 0..train_data.len() / batch_size{
            let batch: (Matrix, Matrix) = NN::batch_to_matrix(&train_data[i*batch_size..i*batch_size+batch_size]);

            train_functions::train_model(self, &batch, &func, step);
            let a = self.loss.clone();
            println!(
                "Batch {} of {} completed  Accuracy: {:5.2}  Cost: {:.4}",
                i+1, n_baches,
                self.categoral_accuracy(train_data, 50),
                self.cost(train_data, 50, &a)
            );
        }
    }

    pub fn print(&self){
        println!("--------------- WEIGHTS ---------------");
        for i in 2..self.layers.len() {
            self.layers[i].weight_matrix.print_dim();
        }
        println!("---------------- BIASES ---------------");
        for i in 1..self.layers.len() {
            self.layers[i].biases.print_dim();
        }
        println!("---------------------------------------");

    }
}