use ndarray::linalg::Dot;
use ndarray::parallel::prelude::*;
use ndarray::{s, Array1, Array2};
use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;
use std::{collections::HashMap, hash::Hash};
use crate::kmeans::KMeans;
use ndarray::{ArrayView1};
use std::collections::{HashSet};
use priority_queue::PriorityQueue;
use ordered_float::OrderedFloat;

pub struct ProductQuantizer {
    sub_vector_dim: usize,
    vector_dim: usize,
    n_clusters: usize,
    codebooks: Vec<Array2<f64>>, // One codebook per subvector
}

impl ProductQuantizer {
    pub fn new(sub_vector_dim: usize, vector_dim: usize, n_clusters: usize) -> Self {
        assert!(
            sub_vector_dim <= vector_dim,
            "sub_vector_dim must be less than or equal to vector_dim"
        );
        assert!(
            vector_dim % sub_vector_dim == 0,
            "vector_dim must be divisible by sub_vector_dim"
        );

        let num_subvectors = vector_dim / sub_vector_dim;
        let codebooks = vec![Array2::zeros((n_clusters, sub_vector_dim)); num_subvectors];

        Self {
            sub_vector_dim,
            vector_dim,
            n_clusters,
            codebooks,
        }
    }

    pub fn train(&mut self, data: &Array2<f64>) {
        let num_subvectors = self.vector_dim / self.sub_vector_dim;

        // Train a separate codebook for each subvector
        for i in 0..num_subvectors {
            // Extract subvector data
            let start_idx = i * self.sub_vector_dim;
            let end_idx = start_idx + self.sub_vector_dim;
            let subvector_data = data.slice(s![.., start_idx..end_idx]).to_owned();

            // Train K-means for this subvector
            let mut kmeans = KMeans::new(self.n_clusters, 100);
            kmeans.fit(&subvector_data);
            self.codebooks[i] = kmeans.centroids().to_owned();
        }
    }

    pub fn encode(&self, data: &Array2<f64>) -> Array2<usize> {
        let (n_samples, _) = data.dim();
        let n_subvectors = self.vector_dim / self.sub_vector_dim;
        let mut encoded = Array2::zeros((n_samples, n_subvectors));

        // For each subvector
        for i in 0..n_subvectors {
            let start_idx = i * self.sub_vector_dim;
            let end_idx = start_idx + self.sub_vector_dim;
            let subvector_data = data.slice(s![.., start_idx..end_idx]);

            // For each sample
            for sample_idx in 0..n_samples {
                let sample = subvector_data.slice(s![sample_idx, ..]);

                // Find closest centroid
                let mut max_sim = f64::NEG_INFINITY;
                let mut closest_idx = 0;

                for (centroid_idx, centroid) in self.codebooks[i].outer_iter().enumerate() {
                    let similarity = sample.dot(&centroid);
                    if similarity > max_sim {
                        max_sim = similarity;
                        closest_idx = centroid_idx;
                    }
                }

                encoded[[sample_idx, i]] = closest_idx;
            }
        }

        encoded
    }

    pub fn decode(&self, data: &Array2<usize>) -> Array2<f64> {
        let (n_samples, n_subvectors) = data.dim();
        let output_dim = n_subvectors * self.sub_vector_dim;
        let mut decoded = Array2::zeros((n_samples, output_dim));

        // For each subvector
        for i in 0..n_subvectors {
            let start_idx = i * self.sub_vector_dim;
            let end_idx = start_idx + self.sub_vector_dim;

            // For each sample
            for sample_idx in 0..n_samples {
                let code = data[[sample_idx, i]];
                let centroid = &self.codebooks[i].slice(s![code, ..]);

                // Copy centroid values to output
                decoded
                    .slice_mut(s![sample_idx, start_idx..end_idx])
                    .assign(&centroid);
            }
        }

        decoded
    }
}

pub struct IVFPQIndex {
    centroids: Array2<f64>,
    inverted_index: HashMap<usize, InvertedList>,
    n_cells: usize,
    vector_dim: usize,
}

struct InvertedList {
    vector_ids: Vec<usize>,
    vectors: Array2<f64>, // Store contiguously in memory
}

impl IVFPQIndex {
    pub fn new(vector_dim: usize, n_cells: usize) -> Self {
        Self {
            centroids: Array2::zeros((0, 0)),
            inverted_index: HashMap::new(),
            n_cells: n_cells,
            vector_dim: vector_dim,
        }
    }

    pub fn train(&mut self, data: &Array2<f64>) {
        // Perform k-means clustering using our custom implementation
        let mut kmeans = KMeans::new(self.n_cells, 1000);
        kmeans.fit(data);
        kmeans.visualize(data, "kmeans.png").unwrap();
        self.centroids = kmeans.centroids().to_owned();

        // Initialize inverted lists
        self.inverted_index = HashMap::with_capacity(self.n_cells);
        self.inverted_index
            .par_extend((0..self.n_cells).into_par_iter().map(|i| {
                (
                    i,
                    InvertedList {
                        vector_ids: Vec::new(),
                        vectors: Array2::zeros((0, self.vector_dim)),
                    },
                )
            }));
    }

    fn find_nearest_cells_batch(
        &self,
        queries: &Array2<f64>,
        n_probes: usize,
    ) -> Vec<Vec<(usize, f64)>> {
        // Compute all dot products at once using matrix multiplication
        let scores_matrix = queries.dot(&self.centroids.t());
        let n_queries = queries.nrows();
        let n_centroids = self.centroids.nrows();

        // Process each query's scores in parallel with pre-allocated vectors
        (0..n_queries)
            .into_par_iter()
            .map(|query_idx| {
                let query_scores = scores_matrix.row(query_idx);
                // Pre-allocate buffer with exact size needed
                let mut scores = Vec::with_capacity(n_centroids);
                // Collect all scores first
                for (idx, &score) in query_scores.iter().enumerate() {
                    scores.push((idx, score));
                }
                // Sort by score in descending order and take top n_probes
                scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                scores.truncate(n_probes);

                scores
            })
            .collect()
    }

    pub fn add(&mut self, vector_ids: &[usize], vectors: &Array2<f64>) {
        // Find nearest cells for all vectors at once
        let nearest_cells_batch = self.find_nearest_cells_batch(vectors, 1);

        // For each vector
        for (i, nearest_cells) in nearest_cells_batch.iter().enumerate() {
            let nearest_cell = nearest_cells[0].0; // Take first cell ID and its index

            // Add vector to the inverted index under its nearest cell
            if let Some(cell) = self.inverted_index.get_mut(&nearest_cell) {
                cell.vector_ids.push(vector_ids[i]);
                cell.vectors.push_row(vectors.slice(s![i, ..])).unwrap();
            }
        }
    }

    pub fn search(
        &self,
        queries: &Array2<f64>,
        k: usize,
        n_probes: usize,
    ) -> Vec<Vec<(usize, f64)>> {
        let nearest_cells_batch = self.find_nearest_cells_batch(queries, n_probes);

        nearest_cells_batch
            .par_iter()
            .enumerate()
            .map(|(query_idx, nearest_cells)| {
                let query = queries.slice(s![query_idx, ..]);
                let mut query_results = Vec::new();

                // Search through cells from highest to lowest score
                for &(cell_id, _) in nearest_cells {
                    if let Some(cell) = self.inverted_index.get(&cell_id) {
                        // Compute dot products with all vectors in this cell at once
                        let scores = cell.vectors.dot(&query);
                        // Add all scores
                        for (i, &score) in scores.iter().enumerate() {
                            query_results.push((cell.vector_ids[i], score));
                        }
                    }
                }

                // Sort and truncate to k results
                if k < 16 {
                    let k = k.min(query_results.len());
                    query_results
                        .select_nth_unstable_by(k, |(_, a), (_, b)| b.partial_cmp(a).unwrap());
                    query_results.truncate(k);
                } else {
                    query_results.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
                    query_results.truncate(k);
                }

                query_results
            })
            .collect()
    }

    pub fn search_conditional(
        &self,
        queries: &Array2<f64>,
        threshold: f64,
        meta_threshold: f64,
    ) -> Vec<Vec<(usize, f64)>> {
        let scores_matrix = queries.dot(&self.centroids.t());
        let n_queries = queries.nrows();

        // Process each query's scores
        let results = (0..n_queries)
            .into_par_iter()
            .map(|query_idx| {
                let centroids_query_ids: Vec<_> = scores_matrix.row(query_idx)
                    .into_iter()
                    .enumerate()
                    .collect();

                let query = queries.slice(s![query_idx, ..]);
                let mut query_results: Vec<(usize, f64)> = Vec::new();
                
                let mut cells_searched = 0;
                let mut total_valid_items = 0;
                let mut total_items = 0;

                // Iterate through centroids in order of similarity
                for (cell_id, &cell_score) in centroids_query_ids {
                    if cell_score < meta_threshold {
                        continue;
                    }
                    cells_searched += 1;
                    
                    if let Some(cell) = self.inverted_index.get(&cell_id) {
                        // Compute dot products with all vectors in this cell
                        let scores = cell.vectors.dot(&query.view());
                        total_items += scores.len();
                        
                        // Add vectors that exceed threshold
                        let valid_items = scores.iter()
                            .enumerate()
                            .filter(|(_, &score)| score > threshold)
                            .map(|(i, &score)| {
                                query_results.push((cell.vector_ids[i], score));
                                1
                            })
                            .sum::<usize>();
                            
                        total_valid_items += valid_items;
                    }
                }

                println!("Query {}: Searched {} cells, average {:.2} valid items per cell, average {:.2} total items per cell", 
                    query_idx, cells_searched, 
                    if cells_searched > 0 { total_valid_items as f64 / cells_searched as f64 } else { 0.0 },
                    if cells_searched > 0 { total_items as f64 / cells_searched as f64 } else { 0.0 }
                );

                query_results
            })
            .collect();

        results
    }
}

pub struct VectorDatabase<T> {
    index: IVFPQIndex,
    values: Vec<T>,
    next_id: usize,
}

impl<T: Clone> VectorDatabase<T> {
    pub fn new(vector_dim: usize, n_cells: usize) -> Self {
        Self {
            index: IVFPQIndex::new(vector_dim, n_cells),
            values: Vec::new(),
            next_id: 0,
        }
    }

    pub fn add(&mut self, vectors: &Array2<f64>, values: &[T]) {
        let n = vectors.nrows();
        let vector_ids: Vec<_> = (self.next_id..self.next_id + n).collect();
        self.index.add(&vector_ids, vectors);
        self.values.extend(values.iter().cloned());
        self.next_id += n;
    }

    pub fn train(&mut self, data: &Array2<f64>) {
        self.index.train(data);
    }

    pub fn search(&self, queries: &Array2<f64>, k: usize, n_probes: usize) -> Vec<Vec<(T, f64)>> {
        let index_results = self.index.search(queries, k, n_probes);        index_results
            .into_iter()
            .map(|results| {
                results
                    .into_iter()
                    .map(|(id, score)| (self.values[id].clone(), score))
                    .collect()
            })
            .collect()
    }

    pub fn search_conditional(
        &self,
        queries: &Array2<f64>,
        threshold: f64,
        meta_threshold: f64,
    ) -> Vec<Vec<(T, f64)>> {
        let start = Instant::now();
        let index_results = self.index.search_conditional(queries, threshold, meta_threshold);        println!("Search took: {:?}", start.elapsed());
        println!("Search took: {:?}", start.elapsed());
        index_results
            .into_iter()
            .map(|results| {
                results
                    .into_iter()
                    .map(|(id, score)| (self.values[id].clone(), score))
                    .collect()
            })
            .collect()
    }
}

pub struct HNSWIndex {
    nodes: Vec<Node>,
    entry_point: Option<usize>,
    ef_construction: usize,
    max_m: usize,
    max_m0: usize,
    max_level: usize,
    level_mult: f64,
}

struct Node {
    vector: Array1<f64>,
    neighbors: Vec<Vec<usize>>,  // Vec of neighbors for each layer
    level: usize,
}

impl HNSWIndex {
    pub fn new(ef_construction: usize, max_m: usize) -> Self {
        Self {
            nodes: Vec::new(),
            entry_point: None,
            ef_construction: ef_construction.max(max_m),
            max_m,
            max_m0: 2 * max_m,
            max_level: 6,
            level_mult: 1.0/ln(max_m as f64),
        }
    }

    fn get_random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        (-r * self.level_mult).floor() as usize
    }

    pub fn add(&mut self, vector: Array1<f64>) {
        let id = self.nodes.len();
        let level = self.get_random_level().min(self.max_level);
        
        // Initialize new node
        let mut node = Node {
            vector,
            neighbors: vec![Vec::new(); level + 1],
            level,
        };

        // If this is the first node
        if self.entry_point.is_none() {
            self.nodes.push(node);
            self.entry_point = Some(0);
            return;
        }

        let mut curr_ep = self.entry_point.unwrap();
        
        // Search from top to bottom layer
        for l in (1..=level).rev() {
            let candidates = self.search_layer(node.vector.view(), curr_ep, self.ef_construction, l);
            curr_ep = self.connect_new_element(&mut node, candidates.into_iter().map(|(id, _)| id).collect(), l);
        }

        // Ground layer (l=0) has more connections
        let candidates = self.search_layer(node.vector.view(), curr_ep, self.ef_construction, 0);
        self.connect_new_element(&mut node, candidates.into_iter().map(|(id, _)| id).collect(), 0);

        // Update entry point if needed
        if level > self.nodes[self.entry_point.unwrap()].level {
            self.entry_point = Some(id);
        }

        self.nodes.push(node);
    }

    fn search_layer_conditional(&self, query: ArrayView1<f64>, entry_point: usize, level: usize, threshold: f64) -> Vec<(usize, f64)> {
        let mut visited = HashSet::new();
        let mut candidates = PriorityQueue::new();
        let mut results = Vec::new();
        
        let dist = cosine_similarity(&self.nodes[entry_point].vector, &query.to_owned());
        candidates.push(entry_point, OrderedFloat(dist));
        if dist > threshold {
            results.push((entry_point, dist));
        }
        visited.insert(entry_point);
        let mut total_neighbors = 0;
        let mut accepted_neighbors = 0;

        while !candidates.is_empty() {
            let (curr, _) = candidates.pop().unwrap();
            
            // Check neighbors at this level
            for &neighbor in &self.nodes[curr].neighbors[level] {
                if !visited.insert(neighbor) {
                    continue;
                }
                total_neighbors += 1;

                let dist = cosine_similarity(&self.nodes[neighbor].vector, &query.to_owned());
                // Add to results only if above threshold
                if dist > threshold {
                    if rand::random::<f64>() < 0.05 {
                        candidates.push(neighbor, OrderedFloat(dist));
                    }
                    results.push((neighbor, dist));
                    accepted_neighbors += 1;
                }
            }
        }

        println!("Total neighbors checked: {}", total_neighbors);
        println!("Neighbors accepted: {}", accepted_neighbors);

        results
    }

    fn search_layer(&self, query: ArrayView1<f64>, entry_point: usize, ef: usize, level: usize) -> Vec<(usize, f64)> {
        let mut visited = HashSet::new();
        let mut candidates = PriorityQueue::new();
        let mut results = HashMap::new();
        let mut result_queue = PriorityQueue::new();
        
        let dist = cosine_similarity(&self.nodes[entry_point].vector, &query.to_owned());
        candidates.push(entry_point, OrderedFloat(dist));
        result_queue.push(entry_point, OrderedFloat(dist));
        results.insert(entry_point, dist);
        visited.insert(entry_point);

        while !candidates.is_empty() {
            let (curr, _) = candidates.pop().unwrap();
            
            // Check neighbors at this level
            for &neighbor in &self.nodes[curr].neighbors[level] {
                if !visited.insert(neighbor) {
                    continue;
                }

                let dist = cosine_similarity(&self.nodes[neighbor].vector, &query.to_owned());
                let worst_dist = result_queue.peek().map(|(_, d)| d.0).unwrap_or(f64::NEG_INFINITY);
                
                if result_queue.len() < ef || dist > worst_dist {
                    candidates.push(neighbor, OrderedFloat(dist));
                    result_queue.push(neighbor, OrderedFloat(dist));
                    results.insert(neighbor, dist);
                    if result_queue.len() > ef {
                        if let Some((removed_id, _)) = result_queue.pop() {
                            results.remove(&removed_id);
                        }
                    }
                }
            }
        }

        result_queue.into_sorted_vec()
        .into_iter()
        .map(|id| (id, results[&id]))
        .collect()
    }

    fn connect_new_element(&mut self, node: &mut Node, candidates: Vec<usize>, level: usize) -> usize {
        let max_connections = if level == 0 { self.max_m0 } else { self.max_m };
        let best_candidates: Vec<usize> = candidates.into_iter()
            .take(max_connections)
            .collect();

        // Add bidirectional connections
        for &candidate_id in &best_candidates {
            node.neighbors[level].push(candidate_id);
            let len = self.nodes.len();
            self.nodes[candidate_id].neighbors[level].push(len);
        }

        best_candidates[0]
    }

    pub fn search(&self, query: &Array1<f64>, k: usize) -> Vec<(usize, f64)> {
        if let Some(ep) = self.entry_point {
            let mut curr_ep = ep;
            let mut curr_level = self.nodes[ep].level;

            // Traverse from top to level 1
            while curr_level > 0 {
                let candidates = self.search_layer(query.view(), curr_ep, self.max_m, curr_level);
                curr_ep = candidates[0].0;
                curr_level -= 1;
            }

            // Search ground layer with ef = max(k, ef_construction)
            self.search_layer(query.view(), curr_ep, k, 0)
        } else {
            Vec::new()
        }
    }

    pub fn search_conditional(&self, query: &Array1<f64>, threshold: f64) -> Vec<(usize, f64)> {
        if let Some(ep) = self.entry_point {
            let mut curr_ep = ep;
            let mut curr_level = self.nodes[ep].level;

            // Traverse from top to level 1
            while curr_level > 0 {
                let candidates = self.search_layer(query.view(), curr_ep, self.max_m, curr_level);
                curr_ep = candidates[0].0;
                curr_level -= 1;
            }

            self.search_layer_conditional(query.view(), curr_ep, 0, threshold)
        } else {
            Vec::new()
        }
    }
}

fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let prod = a.dot(b);
    prod
}

fn ln(x: f64) -> f64 {
    x.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_product_quantizer() {
        let data = Array2::from_shape_fn((128, 128), |_| rand::random::<f64>());
        let mut pq = ProductQuantizer::new(32, 128, 64);
        pq.train(&data);
        println!("{:?}", pq.codebooks[0].shape());
    }

    #[test]
    fn test_encode_decode() {
        // Create test data
        let data = Array2::from_shape_fn((100, 128), |_| rand::random::<f64>());

        // Initialize and train PQ
        let mut pq = ProductQuantizer::new(32, 128, 64);
        pq.train(&data);

        // Encode data
        let encoded = pq.encode(&data);

        // Decode data
        let decoded = pq.decode(&encoded);

        // Check shapes match
        assert_eq!(data.shape(), decoded.shape());

        println!("{:?}", data.slice(s![0, ..]));
        println!("{:?}", decoded.slice(s![0, ..]));

        // Check reconstruction error is reasonable
        let mse = data
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / (data.shape()[0] * data.shape()[1]) as f64;

        // MSE should be relatively small for quantized data
        assert!(mse < 0.1);
    }

    #[test]
    fn test_ivfpq_index() {
        // Create test data with known patterns
        let n_samples = 1000;
        let dim = 128;
        // Create random data and normalize each row
        let mut data = Array2::from_shape_fn((n_samples, dim), |_| rand::random::<f64>());
        for mut row in data.outer_iter_mut() {
            let norm = (row.dot(&row)).sqrt();
            row.mapv_inplace(|x| x / norm);
        }

        // Create normalized test vectors
        let query_vector = Array1::from_vec(vec![1.0 / ((dim as f64).sqrt()); dim]); // normalized to 1
        let similar_vector = Array1::from_vec(vec![0.99 / ((dim as f64).sqrt()); dim]); // normalized to 1
        data.slice_mut(s![0, ..]).assign(&query_vector);
        data.slice_mut(s![1, ..]).assign(&similar_vector);

        // Create and train index
        let n_cells = 32; // Number of Voronoi cells
        let n_probes = 8; // Number of cells to probe during search
        let mut index = IVFPQIndex::new(dim, n_cells);
        index.train(&data);

        // Add vectors in batches
        let batch_size = 100;
        for batch in data
            .outer_iter()
            .enumerate()
            .collect::<Vec<_>>()
            .chunks(batch_size)
        {
            let ids: Vec<usize> = batch.iter().map(|(i, _)| *i).collect();
            let vectors = Array2::from_shape_vec(
                (batch.len(), dim),
                batch
                    .iter()
                    .flat_map(|(_, row)| row.iter().cloned())
                    .collect(),
            )
            .unwrap();
            index.add(&ids, &vectors);
        }

        // Test search with known query
        let k = 5;
        let mut queries = Array2::zeros((2, dim));
        queries.slice_mut(s![0, ..]).assign(&query_vector);
        let results = index.search(&queries, k, n_probes);

        // Validate results
        let first_results = &results[0];
        println!("{:?}", first_results);

        assert!(!first_results.is_empty(), "Search returned no results");
        let (top_id, top_sim) = first_results[0];
        assert!(
            top_sim >= 0.99,
            "Similarity to exact match should be larger than 0.99"
        );
        assert!(top_id == 0, "Top result should be the first vector");

        let (second_id, second_sim) = first_results[1];
        assert!(
            second_sim < top_sim,
            "Similar vector should have lower similarity"
        );
        assert!(second_id == 1, "Second result should be the second vector");
        let results = index.search_conditional(&queries, 10.0, -0.1);
        println!("{:?}", results);
    }

    fn brute_force_search(query: &Array1<f64>, data: &Array2<f64>, k: usize) -> Vec<(usize, f64)> {
        // Compute all dot products at once using matrix multiplication
        let similarities = data.dot(query);

        // Create vector of (index, similarity) pairs
        let mut results: Vec<_> = similarities
            .iter()
            .enumerate()
            .map(|(i, &sim)| (i, sim))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        results
    }

    #[test]
    fn test_ivfpq_accuracy() {
        let mut rng = rand::thread_rng();

        // Create random test data
        let n_samples = 1000;
        let dim = 128;
        let data = Array2::from_shape_fn((n_samples, dim), |_| rng.gen::<f64>());

        // Create and train index
        let n_cells = 16;
        let n_probes = 8;
        println!("\nBuilding IVFPQ index...");
        let build_start = Instant::now();
        let mut index = IVFPQIndex::new(dim, n_cells);
        index.train(&data);

        // Add all vectors
        let ids: Vec<_> = (0..n_samples).collect();
        index.add(&ids, &data);
        let build_time = build_start.elapsed();
        println!("Index build time: {:?}", build_time);

        // Test queries
        let n_queries = 100;
        let k = 10;
        let mut total_recall = 0.0;
        let mut ivfpq_total_time = std::time::Duration::new(0, 0);
        let mut bf_total_time = std::time::Duration::new(0, 0);

        println!("\nAccuracy Test Results:");
        println!("Configuration: {} cells, {} probes", n_cells, n_probes);
        println!("Dataset: {} samples, {} dimensions", n_samples, dim);
        println!(
            "\nTesting {} queries, retrieving top-{} results each",
            n_queries, k
        );

        for i in 0..n_queries {
            // Generate random query
            let query = Array1::from_shape_fn(dim, |_| rng.gen::<f64>());
            let query_matrix = query.clone().into_shape((1, dim)).unwrap();

            // Time IVFPQ search
            let ivfpq_start = Instant::now();
            let ivfpq_results = index.search(&query_matrix, k, n_probes)[0].clone();
            let ivfpq_time = ivfpq_start.elapsed();
            ivfpq_total_time += ivfpq_time;

            let ivfpq_ids: Vec<_> = ivfpq_results.iter().map(|(id, _)| *id).collect();

            // Time brute force search
            let bf_start = Instant::now();
            let bf_results = brute_force_search(&query, &data, k);
            let bf_time = bf_start.elapsed();
            bf_total_time += bf_time;

            let bf_ids: Vec<_> = bf_results.iter().map(|(id, _)| *id).collect();

            // Calculate recall
            let matches: usize = ivfpq_ids.iter().filter(|&id| bf_ids.contains(id)).count();
            let recall = matches as f64 / k as f64;
            total_recall += recall;

            if i < 5 {
                // Print details for first 5 queries
                println!("\nQuery {}:", i);
                println!(
                    "IVFPQ results (took {:?}): {:?}",
                    ivfpq_time,
                    &ivfpq_results[..3]
                );
                println!("Brute results (took {:?}): {:?}", bf_time, &bf_results[..3]);
                println!("Recall: {:.2}%", recall * 100.0);
            }
        }

        let avg_recall = total_recall / n_queries as f64;
        let avg_ivfpq_time = ivfpq_total_time.div_f64(n_queries as f64);
        let avg_bf_time = bf_total_time.div_f64(n_queries as f64);

        println!("\nOverall Results:");
        println!("Average Recall@{}: {:.2}%", k, avg_recall * 100.0);
        println!("Average Search Times:");
        println!("  IVFPQ: {:?} per query", avg_ivfpq_time);
        println!("  Brute Force: {:?} per query", avg_bf_time);
        println!(
            "Speed-up factor: {:.1}x",
            avg_bf_time.as_nanos() as f64 / avg_ivfpq_time.as_nanos() as f64
        );

        assert!(
            avg_recall > 0.3,
            "Recall too low: {:.2}%",
            avg_recall * 100.0
        );
    }

    #[test]
    fn test_vector_database() {
        // Create test data
        let n_samples = 1000;
        let dim = 64;
        let mut rng = rand::thread_rng();
        let data = Array2::from_shape_fn((n_samples, dim), |_| rng.gen::<f64>());

        // Create values to store (using strings as test values)
        let values: Vec<String> = (0..n_samples).map(|i| format!("item_{}", i)).collect();

        // Initialize database
        let n_cells = 32;
        let mut db = VectorDatabase::new(dim, n_cells);

        db.train(&data);

        // Add data in batches
        let batch_size = 100;
        for i in (0..n_samples).step_by(batch_size) {
            let end = std::cmp::min(i + batch_size, n_samples);
            let batch_data = data.slice(s![i..end, ..]).to_owned();
            let batch_values = &values[i..end];
            db.add(&batch_data, batch_values);
        }

        // Test search functionality
        let n_queries = 10;
        let k = 5;
        let n_probes = 8;

        // Create query vectors
        let queries = Array2::from_shape_fn((n_queries, dim), |_| rng.gen::<f64>());

        // Perform search
        let results = db.search(&queries, k, n_probes);

        // Basic validation
        assert_eq!(
            results.len(),
            n_queries,
            "Should return results for each query"
        );
        for result_set in &results {
            assert!(
                result_set.len() <= k,
                "Should return at most k results per query"
            );

            // Check that scores are sorted in descending order
            let scores: Vec<f64> = result_set.iter().map(|(_, score)| *score).collect();
            assert!(
                scores.windows(2).all(|w| w[0] >= w[1]),
                "Results should be sorted by score"
            );
        }

        // Test threshold search
        let threshold = 0.5;
        let threshold_results = db.search_conditional(&queries, threshold, -0.1);
        println!("{:?}", threshold_results);
        // Validate threshold results
        assert_eq!(
            threshold_results.len(),
            n_queries,
            "Should return results for each query"
        );
        for result_set in &threshold_results {
            // Verify all results are above threshold
            assert!(
                result_set.iter().all(|(_, score)| *score > threshold),
                "All results should be above threshold"
            );

            println!("{:?}", result_set);
        }
    }

    #[test]
    fn test_hnsw() {
        let mut rng = rand::thread_rng();
        let n_vectors = 1000;
        let dim = 128;
        let k = 5;

        // Create random vectors for testing
        let vectors: Vec<Array1<f64>> = (0..n_vectors)
            .map(|_| Array1::from_shape_fn(dim, |_| rng.gen::<f64>()))
            .collect();

        // Initialize HNSW index
        let mut hnsw = HNSWIndex::new(128, 16); // ef_construction=128, max_m=16

        // Add vectors to index
        for vector in &vectors {
            hnsw.add(vector.clone());
        }

        // Create query vectors
        let n_queries = 10;
        let queries: Vec<Array1<f64>> = (0..n_queries)
            .map(|_| Array1::from_shape_fn(dim, |_| rng.gen::<f64>()))
            .collect();

        // Perform search
        for query in &queries {
            let results = hnsw.search(query, k);
            println!("{:?}", results);
            // Basic validation
            assert!(
                results.len() <= k,
                "Should return at most k results per query"
            );

            // Verify results by computing actual distances
            let mut actual_distances: Vec<(usize, f64)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, cosine_similarity(v, query)))
                .collect();
            actual_distances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_k_actual: HashSet<_> = actual_distances.iter().take(k * 2).map(|(i, _)| i).collect();
            println!("{:?}", actual_distances.iter().take(k * 2).collect::<Vec<_>>());
            // Check if HNSW results are in top-2k of actual nearest neighbors
            // (allowing some approximation error)
            for &idx in &results {
                assert!(
                    top_k_actual.contains(&idx.0),
                    "HNSW results should be among top-2k actual nearest neighbors"
                );
            }
        }
    }

    #[test]
    fn test_conditional_search() {
        let mut rng = rand::thread_rng();
        let dim = 128;
        let n_vectors = 1024;

        // Create random vectors for testing with both positive and negative values
        let vectors: Vec<Array1<f64>> = (0..n_vectors)
            .map(|_| Array1::from_shape_fn(dim, |_| rng.gen::<f64>() * 2.0 - 1.0))
            .collect();

        // Initialize HNSW index
        let mut hnsw = HNSWIndex::new(320, 16); // ef_construction=128, max_m=16

        // Add vectors to index
        for vector in &vectors {
            hnsw.add(vector.clone());
        }

        // Create query vectors
        let n_queries = 10;
        let queries: Vec<Array1<f64>> = (0..n_queries)
            .map(|_| Array1::from_shape_fn(dim, |_| rng.gen::<f64>()))
            .collect();

        // Test threshold search
        let threshold = 0.4;
        let mut hnsw_total_time = std::time::Duration::new(0, 0);
        let mut ground_truth_total_time = std::time::Duration::new(0, 0);

        for query in &queries {
            // Time HNSW search
            let hnsw_start = std::time::Instant::now();
            let results = hnsw.search_conditional(query, threshold);
            hnsw_total_time += hnsw_start.elapsed();
            
            // Time ground truth calculation
            let ground_truth_start = std::time::Instant::now();
            let ground_truth: Vec<(usize, f64)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, cosine_similarity(v, query)))
                .filter(|(_, sim)| *sim > threshold)
                .collect();
            ground_truth_total_time += ground_truth_start.elapsed();

            // Sort both results and ground truth by similarity for comparison
            let mut results_sorted = results.clone();
            let mut ground_truth_sorted = ground_truth.clone();
            results_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            ground_truth_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Convert to sets for comparison
            let result_set: HashSet<_> = results.into_iter().map(|(idx, _)| idx).collect();
            let ground_truth_set: HashSet<_> = ground_truth.into_iter().map(|(idx, _)| idx).collect();

            // Verify all found results have similarity > threshold
            for &(idx, sim) in &results_sorted {
                assert!(
                    sim > threshold,
                    "All results should have similarity greater than threshold"
                );
                assert!(
                    ground_truth_set.contains(&idx),
                    "All HNSW results should be in ground truth"
                );
            }

            // Print stats
            println!(
                "Query results: found {}/{} vectors above threshold", 
                result_set.len(), 
                ground_truth_set.len()
            );
        }

        // Print timing stats
        println!(
            "Average HNSW search time: {:?}", 
            hnsw_total_time / n_queries as u32
        );
        println!(
            "Average ground truth calculation time: {:?}",
            ground_truth_total_time / n_queries as u32
        );
    }
}
