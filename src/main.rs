mod model;
mod train;
mod index;
mod kmeans;
use clap::{Parser, Subcommand};
use mnist::MnistBuilder;
use ndarray::Array1;
use model::MLP;
use train::{train, TrainingConfig};
use crate::model::DBMLP;
use std::time::Instant;
use ndarray::{Array2, s};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Train {
        #[arg(long, default_value = "50")]
        epochs: usize,
        
        #[arg(long, default_value = "128")]
        batch_size: usize,
        
        #[arg(long, default_value = "0.0001")]
        learning_rate: f64,
        
        #[arg(long, default_value = "8096")]
        hidden_size: usize,
        
        #[arg(long, default_value = "checkpoint.json")]
        checkpoint_path: String,
    },
    Predict {
        #[arg(long, default_value = "checkpoint.json")]
        checkpoint_path: String,
        
        #[arg(long, default_value = "200")]
        num_samples: usize,
        
        #[arg(long, default_value = "false")]
        use_dbmlp: bool,
    },
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train { 
            epochs,
            batch_size,
            learning_rate,
            hidden_size,
            checkpoint_path,
        } => {
            let config = TrainingConfig {
                epochs,
                batch_size,
                learning_rate,
                hidden_size,
                checkpoint_path,
            };
            train(config)
        }
        Commands::Predict { 
            checkpoint_path,
            num_samples,
            use_dbmlp,
        } => {
            // Load model from checkpoint
            let mlp = MLP::from_checkpoint(&checkpoint_path)?;
            
            // Create DBMLP if requested
            let dbmlp = if use_dbmlp {
                Some(DBMLP::new(mlp.clone()))
            } else {
                None
            };
            
            // Load test data
            let mnist = MnistBuilder::new()
                .label_format_digit()
                .training_set_length(50000)
                .validation_set_length(10000)
                .test_set_length(10000)
                .finalize();
            
            println!("Making predictions on {} test samples:", num_samples);
            let mut correct_mlp = 0;
            let mut correct_dbmlp = 0;
            
            let mut total_mlp_time = std::time::Duration::new(0, 0);
            let mut total_dbmlp_time = std::time::Duration::new(0, 0);
            let batch_size = 100;
            let num_batches = (num_samples + batch_size - 1) / batch_size;

            for batch in 0..num_batches {
                let start_idx = batch * batch_size;
                let end_idx = std::cmp::min((batch + 1) * batch_size, num_samples);
                let batch_len = end_idx - start_idx;

                let mut x_batch = Array2::zeros((batch_len, 784));
                for i in 0..batch_len {
                    x_batch.slice_mut(s![i, ..]).assign(&Array1::from_iter(
                        mnist.tst_img[(start_idx + i) * 784..(start_idx + i + 1) * 784]
                            .iter()
                            .map(|&x| f64::from(x) / 255.0)
                    ));
                }

                // Time MLP prediction
                let mlp_start = Instant::now();
                let predicted_mlp = mlp.predict_batch(&x_batch);
                total_mlp_time += mlp_start.elapsed();

                // Time DBMLP prediction
                let predicted_dbmlp = if let Some(ref db) = dbmlp {
                    let dbmlp_start = Instant::now();
                    let pred = db.predict_batch(&x_batch);
                    total_dbmlp_time += dbmlp_start.elapsed();
                    pred
                } else {
                    predicted_mlp.clone()
                };

                // Compare predictions with actual labels
                for i in 0..batch_len {
                    let actual = mnist.tst_lbl[start_idx + i];
                    
                    if predicted_mlp[i] == actual as usize {
                        correct_mlp += 1;
                    }
                    if predicted_dbmlp[i] == actual as usize {
                        correct_dbmlp += 1;
                    }

                    if (start_idx + i) % 100 == 0 {
                        println!(
                            "Sample {}: MLP: {}, DBMLP: {}, Actual: {}", 
                            start_idx + i, predicted_mlp[i], predicted_dbmlp[i], actual
                        );
                    }
                }
            }

            let accuracy_mlp = (correct_mlp as f64) / (num_samples as f64) * 100.0;
            let accuracy_dbmlp = (correct_dbmlp as f64) / (num_samples as f64) * 100.0;
            
            let avg_mlp_time = total_mlp_time.as_millis() as f64 / num_batches as f64;
            let avg_dbmlp_time = if use_dbmlp {
                total_dbmlp_time.as_millis() as f64 / num_batches as f64
            } else {
                0.0
            };
            
            println!("\nResults:");
            println!("MLP Accuracy: {:.2}%, Average time per batch: {:.2} ms", 
                     accuracy_mlp, avg_mlp_time);
            if use_dbmlp {
                println!("DBMLP Accuracy: {:.2}%, Average time per batch: {:.2} ms", 
                         accuracy_dbmlp, avg_dbmlp_time);
                println!("Speed ratio (MLP/DBMLP): {:.2}x", 
                         avg_mlp_time / avg_dbmlp_time);
            }
            Ok(())
        }
    }
}
