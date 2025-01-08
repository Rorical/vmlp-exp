use ndarray::{Array2, ArrayView2, Axis};
use ndarray::linalg::Dot;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;

pub struct KMeans {
    n_clusters: usize,
    max_iterations: usize,
    centroids: Array2<f64>,
}

impl KMeans {
    pub fn new(n_clusters: usize, max_iterations: usize) -> Self {
        Self {
            n_clusters,
            max_iterations,
            centroids: Array2::zeros((0, 0)),
        }
    }

    fn normalize_vector(v: &mut [f64]) {
        let norm_squared: f64 = v.iter().map(|x| x * x).sum();
        if norm_squared > 0.0 {
            let norm = norm_squared.sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
        }
    }

    pub fn fit(&mut self, data: &Array2<f64>) {
        let (n_samples, n_features) = data.dim();
        assert!(n_samples >= self.n_clusters, "Number of samples must be greater than or equal to number of clusters");
        
        // Initialize centroids by randomly selecting k points from the data
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut thread_rng());
        
        // Initialize centroids and normalize them
        self.centroids = Array2::zeros((self.n_clusters, n_features));
        for (i, &idx) in indices.iter().take(self.n_clusters).enumerate() {
            let mut centroid = self.centroids.slice_mut(ndarray::s![i, ..]);
            centroid.assign(&data.slice(ndarray::s![idx, ..]));
            Self::normalize_vector(centroid.as_slice_mut().unwrap());
        }

        let mut prev_centroids = Array2::zeros((self.n_clusters, n_features));
        let tolerance = 1e-6;  // Convergence tolerance
        
        // Main k-means loop
        for iteration in 0..self.max_iterations {
            // Store previous centroids for convergence check
            prev_centroids.assign(&self.centroids);
            
            // Compute assignments using dot product similarity
            let assignments: Vec<usize> = data
                .outer_iter()
                .into_par_iter()
                .map(|sample| {
                    let mut best_cluster = 0;
                    let mut best_similarity = f64::NEG_INFINITY;
                    
                    // Compute dot product with normalized centroids
                    for (cluster_idx, centroid) in self.centroids.outer_iter().enumerate() {
                        let similarity = sample.dot(&centroid);
                        if similarity > best_similarity {
                            best_similarity = similarity;
                            best_cluster = cluster_idx;
                        }
                    }
                    best_cluster
                })
                .collect();

            // Update centroids
            let mut new_centroids = Array2::zeros((self.n_clusters, n_features));
            let mut counts = vec![0usize; self.n_clusters];

            // First accumulate all points (keeping their original magnitudes)
            for (sample_idx, &cluster_idx) in assignments.iter().enumerate() {
                let sample = data.slice(ndarray::s![sample_idx, ..]);
                new_centroids.slice_mut(ndarray::s![cluster_idx, ..])
                    .zip_mut_with(&sample, |a, &b| *a += b);
                counts[cluster_idx] += 1;
            }

            // Then compute means and normalize centroids
            for cluster_idx in 0..self.n_clusters {
                if counts[cluster_idx] > 0 {
                    let mut centroid = new_centroids.slice_mut(ndarray::s![cluster_idx, ..]);
                    // First divide by count to get mean
                    centroid.map_inplace(|x| *x /= counts[cluster_idx] as f64);
                    // Then normalize the centroid
                    Self::normalize_vector(centroid.as_slice_mut().unwrap());
                } else {
                    // If a cluster is empty, reinitialize it with a random point
                    let random_idx = rand::random::<usize>() % n_samples;
                    let mut centroid = new_centroids.slice_mut(ndarray::s![cluster_idx, ..]);
                    centroid.assign(&data.slice(ndarray::s![random_idx, ..]));
                    Self::normalize_vector(centroid.as_slice_mut().unwrap());
                }
            }

            self.centroids = new_centroids;

            // Check for convergence using cosine similarity between old and new centroids
            let mut max_shift = 0.0f64;
            for (old_centroid, new_centroid) in prev_centroids.outer_iter().zip(self.centroids.outer_iter()) {
                let cos_sim = old_centroid.dot(&new_centroid).abs();
                let shift = 1.0 - cos_sim;
                max_shift = f64::max(max_shift, shift);
            }
            
            if max_shift < tolerance && iteration > 0 {
                println!("KMeans converged after {} iterations with max angular shift {}", iteration + 1, max_shift);
                break;
            }
            
            if iteration == self.max_iterations - 1 {
                println!("KMeans reached max iterations ({}) with max angular shift {}", self.max_iterations, max_shift);
            }
        }
    }

    pub fn predict(&self, data: &Array2<f64>) -> Vec<usize> {
        data.outer_iter()
            .into_par_iter()
            .map(|sample| {
                let mut best_cluster = 0;
                let mut best_similarity = f64::NEG_INFINITY;
                
                for (cluster_idx, centroid) in self.centroids.outer_iter().enumerate() {
                    let similarity = sample.dot(&centroid);
                    if similarity > best_similarity {
                        best_similarity = similarity;
                        best_cluster = cluster_idx;
                    }
                }
                best_cluster
            })
            .collect()
    }

    pub fn centroids(&self) -> ArrayView2<f64> {
        self.centroids.view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use rand::Rng;

    #[test]
    fn test_kmeans() {
        let mut rng = rand::thread_rng();
        let n_samples = 1000;
        let dim = 64;
        let n_clusters = 10;
        
        // Generate random test data with varying magnitudes
        let data = Array2::from_shape_fn((n_samples, dim), |_| rng.gen::<f64>() * 10.0);
        
        // Fit KMeans
        let mut kmeans = KMeans::new(n_clusters, 100);
        kmeans.fit(&data);
        
        // Basic checks
        assert_eq!(kmeans.centroids().dim(), (n_clusters, dim));
        
        // Test prediction
        let predictions = kmeans.predict(&data);
        assert_eq!(predictions.len(), n_samples);
        assert!(predictions.iter().all(|&x| x < n_clusters));
        
        // Verify centroids are normalized
        for centroid in kmeans.centroids().outer_iter() {
            let norm: f64 = centroid.iter().map(|x| x * x).sum();
            assert!((norm - 1.0).abs() < 1e-10, "Centroid not normalized: norm = {}", norm);
        }
        
        // Test that points with similar direction but different magnitudes are clustered together
        let mut test_data = Array2::zeros((4, 2));
        test_data.slice_mut(ndarray::s![0, ..]).assign(&Array1::from_vec(vec![1.0, 1.0]));
        test_data.slice_mut(ndarray::s![1, ..]).assign(&Array1::from_vec(vec![2.0, 2.0]));
        test_data.slice_mut(ndarray::s![2, ..]).assign(&Array1::from_vec(vec![1.0, -1.0]));
        test_data.slice_mut(ndarray::s![3, ..]).assign(&Array1::from_vec(vec![2.0, -2.0]));
        
        let mut kmeans = KMeans::new(2, 100);
        kmeans.fit(&test_data);
        let test_predictions = kmeans.predict(&test_data);
        
        // Points with same direction should be in same cluster
        assert_eq!(test_predictions[0], test_predictions[1]);
        assert_eq!(test_predictions[2], test_predictions[3]);
        assert_ne!(test_predictions[0], test_predictions[2]);
    }
} 