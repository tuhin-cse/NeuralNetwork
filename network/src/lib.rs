use matrix::Matrix;
mod activation;

pub use activation::Activation;


#[derive(Debug)]
pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    activation: Activation,
    learning_rate: f64,
}


impl Network {
    pub fn new(layers: Vec<usize>, activation: activation::Activation, learning_rate: f64) -> Network {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }
        Network {
            layers,
            weights,
            biases,
            data: Vec::new(),
            activation,
            learning_rate,
        }
    }

    pub fn feedforward(&mut self, input: Matrix) -> Matrix {
        assert_eq!(self.layers[0], input.data.len(), "Invalid Number of Inputs");
        let mut current = input;
        self.data = vec![current.clone()];
        for i in 0..self.layers.len() - 1 {
            current = self.weights[i].mul(&current).add(&self.biases[i]).map(self.activation.func);
            self.data.push(current.clone());
        }
        current
    }

    pub fn backpropagation(&mut self, output: Matrix, target: Matrix) {
        let mut error = target.sub(&output);
        let mut greadients = output.clone().map(self.activation.derivative);
        for i in (0..self.layers.len() - 1).rev() {
            greadients = greadients.hadamard(&error).mul_scalar(self.learning_rate);
            self.weights[i] = self.weights[i].add(&greadients.mul(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].add(&greadients);
            error = self.weights[i].transpose().mul(&error);
            greadients = self.data[i].map(self.activation.derivative);
        }
    }


    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: usize) {
        for ep in 0..epochs {
            println!("\x1BcEpoch: {}/{}", ep + 1, epochs);
            for i in 0..inputs.len() {
                let input = Matrix::from_array(&*inputs[i].clone());
                let target = Matrix::from_array(&*targets[i].clone());
                let output = self.feedforward(input.clone());
                self.backpropagation(output, target);
            }
        }
    }
}