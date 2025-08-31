use nalgebra::{DVector, DMatrix};

pub enum GradientType {
    Vector(DVector<f64>),
    Matrix(DMatrix<f64>),
}