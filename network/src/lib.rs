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
        self.data.push(input);
        for i in 0..self.layers.len() - 1 {
            let z = self.weights[i].mul(&self.data[i]).add(&self.biases[i]);
            let a = z.map(self.activation.func);
            self.data.push(a);
        }
        self.data[self.data.len() - 1].clone()
    }

    pub fn backpropagation(&mut self, target: Matrix) {
        let mut errors = Vec::new();
        let mut deltas = Vec::new();
        let mut gradients = Vec::new();
        let a = self.data[self.data.len() - 1].clone();
        let error = target.sub(&a);
        let delta = error.map(self.activation.derivative);
        errors.push(error);
        deltas.push(delta);
        for i in (0..self.layers.len() - 1).rev() {
            let a = self.data[i].clone();
            let z = self.weights[i].mul(&a).add(&self.biases[i]);
            let gradient = z.map(self.activation.derivative);
            gradients.push(gradient);
            let delta = deltas[deltas.len() - 1].hadamard(&gradients[gradients.len() - 1]);
            deltas.push(delta.clone());
            let nabla_w = delta.mul(&a.transpose());
            let nabla_b = delta.clone();
            self.weights[i] = self.weights[i].add(&nabla_w.mul_scalar(self.learning_rate));
            self.biases[i] = self.biases[i].add(&nabla_b.mul_scalar(self.learning_rate));
        }
    }
}