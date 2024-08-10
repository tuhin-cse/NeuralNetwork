use network::{Network, Activation};
use matrix::Matrix;

fn main() {
    let layers = vec![2, 3, 3, 1];
    let activation = Activation::sigmoid();
    let learning_rate = 0.1;
    let mut net = Network::new(layers, activation, learning_rate);
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0]
    ];
    let targets = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0]
    ];
    net.train(inputs.clone(), targets.clone(), 5000);
    for input in inputs {
        let output = net.feedforward(Matrix::from_array(&*input));
        println!("Input: {:?}", input);
        println!("Output: {:?}", output.data);
    }
}
