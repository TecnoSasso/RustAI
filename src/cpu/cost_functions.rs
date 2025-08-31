use nalgebra::{DVector};
use crate::cpu::structs::*;
use crate::common::enums::*;

fn mean_squared_error(output: DVector<f64>, expected: &DVector<f64>) -> f64{
    expected.iter()
        .zip(output.iter())
        .map(|(y, p)| (y-p).powi(2))
        .sum()
}

fn cross_entropy(output: DVector<f64>, expected: &DVector<f64>) -> f64{
    if output.min() < 0.0 || output.max() > 1.0 { panic!("Invalid output probability distribution") }

    if expected.min() < 0.0 || expected.max() > 1.0 { panic!("Invalid expected probability distribution") }

    expected.iter()
        .zip(output.iter())
        .map(|(&y, &p)| if y > 0.0 { -y * p.ln() } else { 0.0 })
        .sum()
}

pub fn eval_loss(func: LossFunc, batch: &Vec<(DVector<f64>, DVector<f64>)>, model:&mut NN) -> f64{
    let mut sum: f64 = 0.0;
    for (input, expected) in batch {
        let output: DVector<f64> = model.ff(input, false); 
        sum += match func {
            LossFunc::MeanSquaredError => mean_squared_error(output, expected),
            LossFunc::CrossEntropy => cross_entropy(output, expected),
        }
    }
    sum / batch.len() as f64
}

#[cfg(test)]
mod tests {
    use nalgebra::dvector;
    use super::*;

    #[test]
    fn test_mean_squared_error() {
        assert_eq!(
            mean_squared_error(dvector![], &dvector![]),
            0.0);

        assert!((mean_squared_error(dvector![0.1, 0.2, 0.3], &dvector![0.1, 0.2, 0.4]) - 0.01).abs() < 1e-10);
        assert!((mean_squared_error(dvector![10.0, 0.0, -9.0], &dvector![0.1, 0.2, 0.4]) - 186.41).abs() < 1e-10);
    }

    #[test]
    fn test_cross_entropy() {
        assert_eq!(
            cross_entropy(dvector![], &dvector![]),
            0.0);

        assert_eq!(
            cross_entropy(dvector![0.0, 0.0, 1.0], &dvector![0.0, 0.0, 1.0]),
            0.0);
        
        assert_eq!(
            cross_entropy(dvector![0.1, 0.2, 0.3], &dvector![0.0, 1.0, 0.0]),
            1.6094379124341003);
        assert_eq!(
            cross_entropy(dvector![0.1, 0.2, 0.3], &dvector![0.0, 1.0, 0.0]),
            1.6094379124341003);
    }

    #[test]
    #[should_panic(expected = "Invalid output probability distribution")]
    fn invalid_distr_cross_entropy() {
        cross_entropy(dvector![0.1, -0.2, 0.3], &dvector![0.0, 1.0, 0.0]);
    }
}