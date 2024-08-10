use network::{Network, Activation};

fn main() {
    let layers = vec![2, 3, 1];
    let activation = Activation::sigmoid();
    let learning_rate = 0.1;
    let net = Network::new(layers, activation, learning_rate);
    println!("{:?}", net);
}
