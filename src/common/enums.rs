#[allow(dead_code)]
pub enum Activation {
    ReLu,
    Sigmoid,
    Tanh,
    SoftMax,
}

#[allow(dead_code)]
pub enum LossFunc {
    MeanSquaredError,
    CrossEntropy,
}

impl LossFunc {
    pub fn clone(&mut self) -> Self {
        match self {
            LossFunc::MeanSquaredError => LossFunc::MeanSquaredError,
            LossFunc::CrossEntropy => LossFunc::CrossEntropy
        }
    }
}

#[allow(dead_code)]
pub enum TrainFunc {
    BackProp,
    NEAT,
}