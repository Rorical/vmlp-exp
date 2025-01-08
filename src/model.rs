use ndarray::{Array1, Array2};
use rand_distr::{Normal, Distribution};
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Write, Read};
use crate::index::VectorDatabase;
use std::time::Instant;
#[derive(Serialize, Deserialize)]
pub struct MLPCheckpoint {
    w1: Array2<f64>,
    w2: Array2<f64>,
}

#[derive(Clone)]
pub struct MLP {
    w1: Array2<f64>,
    w2: Array2<f64>,
}

impl MLP {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        
        // He initialization for better training
        let w1_std = (2.0 / input_size as f64).sqrt();
        let w2_std = (2.0 / hidden_size as f64).sqrt();
        
        let normal1 = Normal::new(0.0, w1_std).unwrap();
        let normal2 = Normal::new(0.0, w2_std).unwrap();
        
        let w1 = Array2::from_shape_fn((hidden_size, input_size), |_| {
            normal1.sample(&mut rng)
        });
        let w2 = Array2::from_shape_fn((output_size, hidden_size), |_| {
            normal2.sample(&mut rng)
        });

        MLP { w1, w2 }
    }

    pub fn from_checkpoint(path: &str) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        
        let checkpoint: MLPCheckpoint = serde_json::from_str(&contents)?;
        Ok(MLP {
            w1: checkpoint.w1,
            w2: checkpoint.w2,
        })
    }

    fn relu(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|a| if a > 0.0 { a } else { 0.0 })
    }

    fn relu_derivative(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|a| if a > 0.0 { 1.0 } else { 0.0 })
    }

    pub fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let z1 = self.w1.dot(x);
        let a1 = Self::relu(&z1);
        let z2 = self.w2.dot(&a1);
        (z1, a1, z2)
    }

    pub fn predict(&self, x: &Array1<f64>) -> usize {
        let (_, _, output) = self.forward(x);
        output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0
    }

    pub fn train_step(&mut self, x: &Array1<f64>, y: &Array1<f64>, learning_rate: f64) -> f64 {
        let (z1, a1, z2) = self.forward(x);
        
        let loss = (&z2 - y).mapv(|x| x * x).sum() / (y.len() as f64);
        
        let d_z2 = &z2 - y;
        let d_w2 = d_z2.clone().into_shape((d_z2.len(), 1)).unwrap()
            .dot(&a1.clone().into_shape((1, a1.len())).unwrap());
        
        let d_a1 = self.w2.t().dot(&d_z2);
        let d_z1 = &d_a1 * &Self::relu_derivative(&z1);
        let d_w1 = d_z1.clone().into_shape((d_z1.len(), 1)).unwrap()
            .dot(&x.clone().into_shape((1, x.len())).unwrap());
        
        self.w2 = &self.w2 - learning_rate * &d_w2;
        self.w1 = &self.w1 - learning_rate * &d_w1;
        
        loss
    }

    pub fn save_checkpoint(&self, path: &str) -> std::io::Result<()> {
        let checkpoint = MLPCheckpoint {
            w1: self.w1.clone(),
            w2: self.w2.clone(),
        };

        let json = serde_json::to_string_pretty(&checkpoint)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn train_batch(&mut self, x_batch: &Array2<f64>, y_batch: &Array2<f64>, learning_rate: f64) -> f64 {
        let batch_size = x_batch.nrows();
        
        // Forward pass
        let z1 = x_batch.dot(&self.w1.t());
        let a1 = z1.mapv(|x| if x > 0.0 { x } else { 0.0 }); // ReLU
        let z2 = a1.dot(&self.w2.t());
        
        // Compute loss
        let loss = (&z2 - y_batch).mapv(|x| x * x).sum() / (batch_size as f64);
        
        // Backward pass
        let d_z2 = &z2 - y_batch;
        let d_w2 = d_z2.t().dot(&a1) / (batch_size as f64);
        
        let d_a1 = d_z2.dot(&self.w2);
        let d_z1 = &d_a1 * &z1.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }); // ReLU derivative
        let d_w1 = d_z1.t().dot(x_batch) / (batch_size as f64);
        
        // Update weights with momentum
        self.w2 = &self.w2 - learning_rate * &d_w2;
        self.w1 = &self.w1 - learning_rate * &d_w1;
        
        loss
    }
    
    pub fn predict_batch(&self, x_batch: &Array2<f64>) -> Vec<usize> {
        let z1 = x_batch.dot(&self.w1.t());
        let a1 = z1.mapv(|x| if x > 0.0 { x } else { 0.0 });
        let z2 = a1.dot(&self.w2.t());
        
        z2.outer_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0
            })
            .collect()
    }
}

pub struct DBMLP {
    db: VectorDatabase<Array1<f64>>,
}

impl DBMLP {
    pub fn new(mlp: MLP) -> Self {
        let mut db = VectorDatabase::new(mlp.w1.ncols(), 128);
        let mut keys = mlp.w1.to_owned();
        for mut row in keys.outer_iter_mut() {
            let norm_squared: f64 = row.iter().map(|x| x * x).sum();
            if norm_squared > 0.0 {
                let norm = norm_squared.sqrt();
                row.mapv_inplace(|x| x / norm);
            }
        }
        db.train(&keys);
        let values: Vec<_> = (0..mlp.w2.ncols()).map(|i| mlp.w2.column(i).to_owned()).collect();
        db.add(&keys, &values);
        DBMLP { db }
    }

    pub fn predict_batch(&self, x_batch: &Array2<f64>) -> Vec<usize> {
        let start = Instant::now();
        let results = self.db.search_conditional(x_batch, 0.0, -0.01);
        println!("DB Search took: {:?}", start.elapsed());

        results.iter().map(|query_results| {
            // Compute weighted sums for each class (0-9) using dot products
            let mut sums = vec![0.0; 10];
            for (value, score) in query_results {
                for i in 0..10 {
                    sums[i] += value[i] * score;
                }
            }

            // Return index of maximum sum for this query
            sums.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0
        }).collect()
    }
}
