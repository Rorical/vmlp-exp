use crate::model::MLP;
use mnist::MnistBuilder;
use ndarray::{Array1, Array2};

pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub hidden_size: usize,
    pub checkpoint_path: String,
}

pub fn train(config: TrainingConfig) -> std::io::Result<()> {
    // Load MNIST dataset
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50000)
        .validation_set_length(10000)
        .test_set_length(10000)
        .finalize();

    let train_data = mnist.trn_img;
    let train_labels = mnist.trn_lbl;
    let test_data = mnist.tst_img;
    let test_labels = mnist.tst_lbl;

    // Initialize MLP
    let mut mlp = MLP::new(784, config.hidden_size, 10);
    
    let num_samples = train_data.len() / 784;
    let num_batches = num_samples / config.batch_size;
    let mut best_accuracy = 0.0;

    // Training loop
    for epoch in 0..config.epochs {
        let mut total_loss = 0.0;

        // Process data in batches
        for batch in 0..num_batches {
            let start_idx = batch * config.batch_size;
            let end_idx = start_idx + config.batch_size;
            
            // Prepare batch input data (normalize to 0-1 range)
            let mut x_batch = Array2::zeros((config.batch_size, 784));
            let mut y_batch = Array2::zeros((config.batch_size, 10));
            
            // Fill batch arrays
            for (i, sample_idx) in (start_idx..end_idx).enumerate() {
                // Input data
                let x_slice = &train_data[sample_idx * 784..(sample_idx + 1) * 784];
                x_batch.row_mut(i).assign(&Array1::from_iter(
                    x_slice.iter().map(|&x| f64::from(x) / 255.0)
                ));
                
                // Target (one-hot encoding)
                y_batch[[i, train_labels[sample_idx] as usize]] = 1.0;
            }
            
            // Train on batch
            total_loss += mlp.train_batch(&x_batch, &y_batch, config.learning_rate);
        }
        
        let epoch_loss = total_loss / num_batches as f64;
        println!("Epoch {}, Average Loss: {}", epoch, epoch_loss);
        
        // Evaluate on test set
        if epoch % 1 == 0 {
            let test_samples = 1000;
            let num_test_batches = test_samples / config.batch_size;
            let mut correct = 0;
            
            // Evaluate in batches
            for batch in 0..num_test_batches {
                let start_idx = batch * config.batch_size;
                let end_idx = start_idx + config.batch_size;
                
                // Prepare test batch
                let mut x_test_batch = Array2::zeros((config.batch_size, 784));
                for (i, sample_idx) in (start_idx..end_idx).enumerate() {
                    let x_slice = &test_data[sample_idx * 784..(sample_idx + 1) * 784];
                    x_test_batch.row_mut(i).assign(&Array1::from_iter(
                        x_slice.iter().map(|&x| f64::from(x) / 255.0)
                    ));
                }
                
                // Get predictions for batch
                let predictions = mlp.predict_batch(&x_test_batch);
                
                // Count correct predictions
                for (i, &pred) in predictions.iter().enumerate() {
                    if pred == test_labels[start_idx + i] as usize {
                        correct += 1;
                    }
                }
            }
            
            let accuracy = 100.0 * correct as f64 / (num_test_batches * config.batch_size) as f64;
            println!("Test Accuracy: {:.2}%", accuracy);

            if accuracy > best_accuracy {
                best_accuracy = accuracy;
                mlp.save_checkpoint(&config.checkpoint_path)?;
                println!("Saved checkpoint with accuracy: {:.2}%", accuracy);
            }
        }
    }

    Ok(())
} 