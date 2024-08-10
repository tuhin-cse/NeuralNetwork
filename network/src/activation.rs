#[derive(Debug)]
pub struct Activation {
    pub func: fn(f64) -> f64,
    pub derivative: fn(f64) -> f64,
}

impl Activation {
    pub fn sigmoid() -> Activation {
        Activation {
            func: |x| 1.0 / (1.0 + (-x).exp()),
            derivative: |y| y * (1.0 - y),
        }
    }

    pub fn tanh() -> Activation {
        Activation {
            func: |x| x.tanh(),
            derivative: |y| 1.0 - y.powi(2),
        }
    }

    pub fn relu() -> Activation {
        Activation {
            func: |x| if x > 0.0 { x } else { 0.0 },
            derivative: |y| if y > 0.0 { 1.0 } else { 0.0 },
        }
    }

    pub fn leaky_relu() -> Activation {
        Activation {
            func: |x| if x > 0.0 { x } else { 0.01 * x },
            derivative: |y| if y > 0.0 { 1.0 } else { 0.01 },
        }
    }

    pub fn softmax() -> Activation {
        Activation {
            func: |x| x.exp(),
            derivative: |y| y * (1.0 - y),
        }
    }
}