use ndarray::{Array2, ArrayView2, Axis};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use plotters::prelude::full_palette::*;


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
            //Self::normalize_vector(centroid.as_slice_mut().unwrap());
        }

        let mut prev_centroids = Array2::zeros((self.n_clusters, n_features));
        let tolerance = 1e-4;  // Convergence tolerance
        
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
                    //Self::normalize_vector(centroid.as_slice_mut().unwrap());
                } else {
                    // If a cluster is empty, reinitialize it with a random point
                    let random_idx = rand::random::<usize>() % n_samples;
                    let mut centroid = new_centroids.slice_mut(ndarray::s![cluster_idx, ..]);
                    centroid.assign(&data.slice(ndarray::s![random_idx, ..]));
                    //Self::normalize_vector(centroid.as_slice_mut().unwrap());
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

    /// Visualizes the clusters for 2D data using plotters.
    /// Returns an error if the data is not 2-dimensional.
    pub fn visualize(&self, data: &Array2<f64>, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use plotters::prelude::*;

        // Verify data is 2D
        let (n_samples, n_features) = data.dim();

        // If data is not 2D, reduce it using PCA
        let data_2d = if n_features != 2 {
            // Center the data
            let mean = data.mean_axis(Axis(0)).unwrap();
            let centered = data - &mean.view().insert_axis(Axis(0));
            
            // Compute covariance matrix
            let cov = centered.t().dot(&centered) / (n_samples as f64 - 1.0);
            
            // Power iteration to find top 2 eigenvectors
            let mut eigenvectors = Array2::from_shape_fn((n_features, 2), |_| rand::random::<f64>());
            
            for _ in 0..50 {  // Number of iterations
                // Normalize columns
                for mut col in eigenvectors.columns_mut() {
                    let norm = (col.dot(&col)).sqrt();
                    col.map_inplace(|x| *x /= norm);
                }
                
                // Power iteration step
                let new_vectors = cov.dot(&eigenvectors);
                
                // Gram-Schmidt orthogonalization
                let mut v1 = new_vectors.column(0).to_owned();
                let norm1 = (v1.dot(&v1)).sqrt();
                v1.map_inplace(|x| *x /= norm1);
                
                let mut v2 = new_vectors.column(1).to_owned();
                let proj = v1.dot(&v2);
                v2 = &v2 - &(&v1 * proj);
                let norm2 = (v2.dot(&v2)).sqrt();
                v2.map_inplace(|x| *x /= norm2);
                
                eigenvectors.column_mut(0).assign(&v1);
                eigenvectors.column_mut(1).assign(&v2);
            }
            
            // Project both data and centroids
            let centered_centroids = &self.centroids - &mean.view().insert_axis(Axis(0));
            (centered.dot(&eigenvectors), centered_centroids.dot(&eigenvectors))
        } else {
            (data.to_owned(), self.centroids.to_owned())
        };

        // Get cluster assignments
        let assignments = self.predict(data);

        // Unpack the results
        let (data_2d, centroids_2d) = data_2d;

        // Create output file
        let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        // Find data bounds
        let x_min = data_2d.column(0).fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = data_2d.column(0).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_min = data_2d.column(1).fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = data_2d.column(1).fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Add some padding to the bounds
        let padding = 0.1 * ((x_max - x_min).max(y_max - y_min));
        
        let mut chart = ChartBuilder::on(&root)
            .caption("KMeans Clustering", ("sans-serif", 30))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                (x_min - padding)..(x_max + padding),
                (y_min - padding)..(y_max + padding),
            )?;

        chart.configure_mesh().draw()?;

        // Define colors for different clusters
        let colors = [
            &RED, &BLUE, &GREEN, &CYAN, &MAGENTA, &YELLOW,
            &BLACK, &WHITE, &GREY, &PURPLE,
        ];

        // Plot data points
        for cluster_idx in 0..self.n_clusters {
            let cluster_points: Vec<(f64, f64)> = assignments
                .iter()
                .zip(data_2d.outer_iter())
                .filter(|(&c, _)| c == cluster_idx)
                .map(|(_, row)| (row[0], row[1]))
                .collect();

            chart.draw_series(
                cluster_points
                    .iter()
                    .map(|point| Circle::new(*point, 3, colors[cluster_idx % colors.len()].filled())),
            )?;
        }

        // Plot centroids with larger size and border
        chart.draw_series(
            centroids_2d
                .outer_iter()
                .enumerate()
                .map(|(i, centroid)| {
                    Circle::new(
                        (centroid[0], centroid[1]),
                        16,
                        colors[i % colors.len()].filled().stroke_width(2),
                    )
                }),
        )?;

        root.present()?;
        Ok(())
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
        let n_samples = 200;
        let dim = 8;
        let n_clusters = 4;
        
        // Generate four distinct groups of data
        let mut data = Array2::zeros((n_samples, dim));
        
        // Group 1: Cluster around (3, 3)
        for i in 0..(n_samples/4) {
            data[[i, 0]] = 3.0 + rng.gen::<f64>() - 0.5; // x coordinate
            data[[i, 1]] = 3.0 + rng.gen::<f64>() - 0.5; // y coordinate
        }
        
        // Group 2: Cluster around (-3, 3) 
        for i in (n_samples/4)..(n_samples/2) {
            data[[i, 0]] = -3.0 + rng.gen::<f64>() - 0.5;
            data[[i, 1]] = 3.0 + rng.gen::<f64>() - 0.5;
        }
        
        // Group 3: Cluster around (-3, -3)
        for i in (n_samples/2)..(3*n_samples/4) {
            data[[i, 0]] = -3.0 + rng.gen::<f64>() - 0.5;
            data[[i, 1]] = -3.0 + rng.gen::<f64>() - 0.5;
        }
        
        // Group 4: Cluster around (3, -3)
        for i in (3*n_samples/4)..n_samples {
            data[[i, 0]] = 3.0 + rng.gen::<f64>() - 0.5;
            data[[i, 1]] = -3.0 + rng.gen::<f64>() - 0.5;
        }
        // Fit KMeans
        let mut kmeans = KMeans::new(n_clusters, 100);
        kmeans.fit(&data);
        
        // Basic checks
        assert_eq!(kmeans.centroids().dim(), (n_clusters, dim));
        
        // Test prediction
        let predictions = kmeans.predict(&data);
        assert_eq!(predictions.len(), n_samples);
        assert!(predictions.iter().all(|&x| x < n_clusters));

        // Visualize clusters
        kmeans.visualize(&data, "kmeans_clusters.png").unwrap();
        
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