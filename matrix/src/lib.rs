#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}


impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }

    pub fn print(&self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                print!("{} ", self.get(i, j));
            }
            println!();
        }
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j) + other.get(i, j));
            }
        }
        result
    }

    pub fn sub(&self, other: &Matrix) -> Matrix {
        if(self.rows != other.rows || self.cols != other.cols) {
            panic!("Matrices must have the same dimensions");
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j) - other.get(i, j));
            }
        }
        result
    }

    pub fn mul(&self, other: &Matrix) -> Matrix {
        if(self.cols != other.rows) {
            panic!("Columns of A must match rows of B");
        }
        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    pub fn mul_scalar(&self, scalar: f64) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j) * scalar);
            }
        }
        result
    }

    pub fn hadamard(&self, other: &Matrix) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j) * other.get(i, j));
            }
        }
        result
    }

    pub fn map(&self, f: fn(f64) -> f64) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, f(self.get(i, j)));
            }
        }
        result
    }

    pub fn static_map(matrix: &Matrix, f: fn(f64) -> f64) -> Matrix {
        let mut result = Matrix::new(matrix.rows, matrix.cols);
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                result.set(i, j, f(matrix.get(i, j)));
            }
        }
        result
    }

    pub fn from_array(arr: &[f64]) -> Matrix {
        let mut result = Matrix::new(arr.len(), 1);
        for i in 0..arr.len() {
            result.set(i, 0, arr[i]);
        }
        result
    }

    pub fn to_array(&self) -> Vec<f64> {
        let mut result = Vec::new();
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.push(self.get(i, j));
            }
        }
        result
    }

    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut result = Matrix::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                result.set(i, j, rand::random::<f64>() * 2.0 - 1.0);
            }
        }
        result
    }

    pub fn randomize(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.set(i, j, rand::random::<f64>() * 2.0 - 1.0);
            }
        }
    }
}


// macro_rules! matrix {
//     ($($x:expr),*) => {
//         {
//             let mut matrix = Matrix::new(1, 0);
//             $(
//                 matrix.data.push($x);
//                 matrix.cols += 1;
//             )*
//             matrix.rows = 1;
//             matrix
//         }
//     };
// }