use crate::algorithms::vamana::VamanaError;
use crate::bgworker::storage::Storage;
use crate::bgworker::storage::StoragePreallocator;
use crate::bgworker::storage_mmap::MmapBox;
use crate::bgworker::vectors::Vectors;
use crate::prelude::*;

use rand::distributions::Uniform;
use rand::prelude::SliceRandom;
use rand::Rng;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashSet};
use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;


pub struct VertexWithDistance {
    pub id: usize,
    pub distance: Scalar,
}

impl VertexWithDistance {
    pub fn new(id: usize, distance: Scalar) -> Self {
        Self { id, distance }
    }
}

impl PartialEq for VertexWithDistance {
    fn eq(&self, other: &Self) -> bool {
        self.distance.eq(&other.distance)
    }
}

impl Eq for VertexWithDistance {}

impl PartialOrd for VertexWithDistance {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.distance.cmp(&other.distance))
    }
}

impl Ord for VertexWithDistance {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.cmp(&other.distance)
    }
}

/// DiskANN search state.
pub struct SearchState {
    pub visited: HashSet<usize>,
    candidates: BTreeMap<Scalar, usize>,
    heap: BinaryHeap<Reverse<VertexWithDistance>>,
    heap_visited: HashSet<usize>,
    l: usize,
    /// Number of results to return.
    //TODO: used during search.
    #[allow(dead_code)]
    k: usize,
}

impl SearchState {
    /// Creates a new search state.
    pub(crate) fn new(k: usize, l: usize) -> Self {
        Self {
            visited: HashSet::new(),
            candidates: BTreeMap::new(),
            heap: BinaryHeap::new(),
            heap_visited: HashSet::new(),
            k,
            l,
        }
    }

    /// Return the next unvisited vertex.
    fn pop(&mut self) -> Option<usize> {
        while let Some(vertex) = self.heap.pop() {
            if !self.candidates.contains_key(&vertex.0.distance) {
                // The vertex has been removed from the candidate lists,
                // from [`push()`].
                continue;
            }

            self.visited.insert(vertex.0.id);
            return Some(vertex.0.id);
        }

        None
    }

    /// Push a new (unvisited) vertex into the search state.
    fn push(&mut self, vertex_id: usize, distance: Scalar) {
        assert!(!self.visited.contains(&vertex_id));
        self.heap_visited.insert(vertex_id);
        self.heap
            .push(Reverse(VertexWithDistance::new(vertex_id, distance)));
        self.candidates.insert(distance, vertex_id);
        if self.candidates.len() > self.l {
            self.candidates.pop_last();
        }
    }

    /// Mark a vertex as visited.
    fn visit(&mut self, vertex_id: usize) {
        self.visited.insert(vertex_id);
    }

    // Returns true if the vertex has been visited.
    fn is_visited(&self, vertex_id: usize) -> bool {
        self.visited.contains(&vertex_id) || self.heap_visited.contains(&vertex_id)
    }
}

#[async_trait]
pub trait AsyncFunc {
    /// Distance between two vertices, specified by their IDs.
    async fn distance(&self, a: usize, b: usize) -> Result<Scalar>;

    /// Distance from query vector to a vertex identified by the idx.
    async fn distance_to(&self, query: &[Scalar], idx: usize) -> Result<Scalar>;

    /// Return the neighbor
    async fn neighbors(&self, id: usize) -> Result<Arc<[usize]>>;
}

#[allow(unused)]
pub struct VamanaImpl<D: DistanceFamily> {
    neighbors: MmapBox<[usize]>,
    neighbor_size: MmapBox<[usize]>,
    vectors: Arc<Vectors>,
    dims: u16,
    r: usize,
    alpha: f32,
    l: usize,
    _maker: PhantomData<D>,
}

unsafe impl<D: DistanceFamily> Send for VamanaImpl<D> {}
unsafe impl<D: DistanceFamily> Sync for VamanaImpl<D> {}

#[async_trait]
impl<D: DistanceFamily> AsyncFunc for VamanaImpl<D> {
    async fn distance(&self, a: usize, b: usize) -> Result<Scalar>{
        Ok(D::distance(self.vectors.get_vector(a), self.vectors.get_vector(b)));
    }

    async fn distance_to(&self, query: &[Scalar], idx: usize) -> Result<Scalar>{
        Ok(D::distance(query, self.vectors.get_vector(idx)));
    }

    async fn get_neighbors(&self, id: usize) -> Result<Arc<[usize]>>{
        let size = self.neighbor_size[id];
        Ok(Arc::new(self.neighbors[(id * self.r)..(id * self.r + size)]))
    }

    async fn greedy_search(
        &self,
        start: usize,
        query: &[Scalar],
        k: usize,
        search_size: usize,
    ) -> Result<SearchState, VamanaError> {
        let mut state = SearchState::new(k, search_size);

        let dist = self.distance_to(query, start).await?;
        state.push(start, dist);
        while let Some(id) = state.pop() {
            // only pop id in the search list but not visited
            state.visit(id);

            let neighbor_ids = self.get_neighbors(id).await?;
            for &neighbor_id in neighbor_ids {
                if state.is_visited(neighbor_id) {
                    continue;
                }

                let dist = self.distance_to(query, neighbor_id).await?;
                state.push(neighbor_id, dist); // push and retain closet l nodes
            }
        }

        Ok(state)
    }

    async fn robust_prune(
        &self,
        id: usize,
        mut visited: HashSet<usize>,
        alpha: f32,
        r: usize,
    ) -> Result<Vec<usize>, VamanaError> {
        visited.remove(&id); // in case visited has id itself
        let neighbor_ids = self.get_neighbors(id).await?;
        visited.extend(neighbor_ids.iter());

        let mut heap: BinaryHeap<VertexWithDistance> = visited
            .iter()
            .map(|v| {
                let dist = D::distance(self.vectors.get_vector(id), self.vectors.get_vector(*v));
                VertexWithDistance {
                    id: *v,
                    distance: dist,
                }
            })
            .collect();

        let new_neighbor_ids = tokio::task::spawn_blocking( move || {
            let mut new_neighbor_ids: Vec<usize> = vec![];
            while !visited.is_empty() {
                let mut p = heap.pop().unwrap();
                while !visited.contains(&p.id) {
                    p = heap.pop().unwrap();
                }
    
                new_neighbor_ids.push(p.id);
                if new_neighbor_ids.len() >= r {
                    break;
                }
                let mut to_remove: HashSet<usize> = HashSet::new();
                for pv in visited.iter() {
                    let dist_prime =
                        D::distance(self.vectors.get_vector(p.id), self.vectors.get_vector(*pv));
                    let dist_query =
                        D::distance(self.vectors.get_vector(id), self.vectors.get_vector(*pv));
                    if Scalar::from(alpha) * dist_prime <= dist_query {
                        to_remove.insert(*pv);
                    }
                }
                for pv in to_remove.iter() {
                    visited.remove(pv);
                }
            }
            Ok::<_, VamanaError>(new_neighbours)
        })
        .await??
        Ok(new_neighbor_ids)
    }

    async fn one_pass(
        &mut self,
        medoid: usize,
        alpha: f32,
        r: usize,
        l: usize,
        mut rng: impl Rng,
    ) -> Result<(), VamanaError> {
        let len = self.vectors.len();
        let mut ids = (0..len).collect::<Vec<_>>();
        ids.shuffle(&mut rng);

        for &id in ids.iter() {
            let query = self.vectors.get_vector(id);
            let state = self.greedy_search(medoid, query, 1, l).await?;
            let neighbor_ids = self.robust_prune(id, state.visited, alpha, l).await?;
            let neighbor_ids: HashSet<usize> = neighbor_ids.iter().cloned().collect();

            self._set_neighbors(id, &neighbor_ids);

            let neighbor_ids = stream::iter(neighbor_ids)
            .map(|j| async move {
                let mut old_neighbors: HashSet<usize> = 
                    self.get_neighbors(j)
                    .await?
                    .values()
                    .iter()
                    .map(|v| *v as usize)
                    .collect();
                old_neighbors.insert(id);
                if old_neighbors.len() + 1 > r {
                    let new_neighbors = self._robust_prune(neighbor_id, old_neighbors, alpha, r).await?;
                    Ok::<_, Error>((j as usize, new_neighbours))
                } else {
                    Ok::<_, Error>((
                        j as usize,
                        neighbor_set.iter().map(|n| *n as u32).collect::<Vec<_>>(),
                    ))
                }
            })
            .buffered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?;

            for &neighbor_id in neighbor_ids.iter() {
                let old_neighbors = self.get_neighbors(neighbor_id).await?;
                let mut old_neighbors: HashSet<usize> = old_neighbors.iter().cloned().collect();
                old_neighbors.insert(id);
                if old_neighbors.len() > r {
                    // need robust prune
                    let new_neighbors = self._robust_prune(neighbor_id, old_neighbors, alpha, r)?;
                    let new_neighbors: HashSet<usize> = new_neighbors.iter().cloned().collect();
                    self._set_neighbors(neighbor_id, &new_neighbors);
                } else {
                    self._set_neighbors(neighbor_id, &old_neighbors);
                }
            }
        }

        Ok(())
    }

}

impl<D: DistanceFamily> VamanaImpl<D> {
    pub fn prebuild(
        storage: &mut StoragePreallocator,
        capacity: usize,
        r: usize,
        memmap: Memmap,
    ) -> Result<(), VamanaError> {
        let number_of_nodes = capacity;
        storage.palloc_mmap_slice::<usize>(memmap, r * number_of_nodes);
        storage.palloc_mmap_slice::<usize>(memmap, number_of_nodes);
        Ok(())
    }

    pub fn new(
        storage: &mut Storage,
        vectors: Arc<Vectors>,
        n: usize,
        dims: u16,
        r: usize,
        alpha: f32,
        l: usize,
        memmap: Memmap,
    ) -> Result<Self, VamanaError> {
        assert!(n == vectors.len());
        let number_of_nodes = n;
        let neighbors = unsafe {
            storage
                .alloc_mmap_slice::<usize>(memmap, r * number_of_nodes)
                .assume_init()
        };
        let neighbor_size = unsafe {
            storage
                .alloc_mmap_slice::<usize>(memmap, number_of_nodes)
                .assume_init()
        };

        let mut new_vamana = Self {
            neighbors,
            neighbor_size,
            vectors: vectors.clone(),
            dims,
            r,
            alpha,
            l,
            _maker: PhantomData,
        };

        // 1. init graph with r random neighbors for each node
        let rng = rand::thread_rng();
        let len = vectors.len();
        new_vamana._init_graph(len, rng.clone());

        // 2. find medoid
        let medoid = new_vamana._find_medoid();

        // 3. iterate pass (TODO: lancedb use two passes here, need further investigation)
        new_vamana._one_pass(medoid, alpha, r, l, rng.clone())?;

        Ok(new_vamana)
    }

    pub fn load(
        storage: &mut Storage,
        vectors: Arc<Vectors>,
        capacity: usize,
        dims: u16,
        r: usize,
        alpha: f32,
        l: usize,
        memmap: Memmap,
    ) -> Result<Self, VamanaError> {
        let number_of_nodes = capacity;
        Ok(Self {
            neighbors: unsafe {
                storage
                    .alloc_mmap_slice(memmap, r * number_of_nodes)
                    .assume_init()
            },
            neighbor_size: unsafe {
                storage
                    .alloc_mmap_slice(memmap, number_of_nodes)
                    .assume_init()
            },
            vectors: vectors,
            dims: dims,
            r: r,
            alpha: alpha,
            l: l,
            _maker: PhantomData,
        })
    }

    #[allow(unused)]
    pub fn search<F>(
        &self,
        target: Box<[Scalar]>,
        k: usize,
        filter: F,
    ) -> Result<Vec<(Scalar, u64)>, VamanaError>
    where
        F: FnMut(u64) -> bool,
    {
        // TODO: filter
        let state = self._greedy_search(0, &target, k, k * 2)?;

        let mut results = BinaryHeap::<(Scalar, usize)>::new();
        for (distance, row) in state.candidates {
            if results.len() == k {
                break;
            }

            results.push((Scalar::from(distance), row));
        }
        let res_vec: Vec<(Scalar, u64)> = results
            .iter()
            .map(|x| (x.0, self.vectors.get_data(x.1)))
            .collect();
        Ok(res_vec)
    }

    #[allow(unused)]
    pub fn insert(&self, x: usize) -> Result<(), VamanaError> {
        // TODO: the insert API is a fake insert for user,
        // but can be used to implement concurrent index building
        Ok(())
    }

    fn _init_graph(&mut self, n: usize, mut rng: impl Rng) {
        let distribution = Uniform::new(0, n);
        for i in 0..n {
            let mut neighbor_ids: HashSet<usize> = HashSet::new();
            while neighbor_ids.len() < self.r {
                let neighbor_id = rng.sample(distribution);
                if neighbor_id != i {
                    neighbor_ids.insert(neighbor_id);
                }
            }

            self._set_neighbors(i, &neighbor_ids);
        }
    }

    fn _set_neighbors(&mut self, vertex_index: usize, neighbor_ids: &HashSet<usize>) {
        assert!(neighbor_ids.len() <= self.r);
        let mut i = 0;
        for item in neighbor_ids {
            self.neighbors[vertex_index * self.r + i] = *item;
            i += 1;
        }
        self.neighbor_size[vertex_index] = neighbor_ids.len();
    }

    fn _get_neighbors(&self, vertex_index: usize) -> &[usize] {
        //TODO: store neighbor length
        let size = self.neighbor_size[vertex_index];
        &self.neighbors[(vertex_index * self.r)..(vertex_index * self.r + size)]
    }

    fn _find_medoid(&self) -> usize {
        // TODO: batch and concurrent
        let centroid = self._compute_centroid();
        let centroid_arr: &[Scalar] = &centroid;

        let len = self.vectors.len();
        let mut medoid_index = 0;
        let mut min_dis = Scalar::INFINITY;
        for i in 0..len {
            let dis = D::distance(centroid_arr, self.vectors.get_vector(i));
            if dis < min_dis {
                min_dis = dis;
                medoid_index = i;
            }
        }
        medoid_index
    }

    fn _compute_centroid(&self) -> Vec<Scalar> {
        // TODO: batch and concurrent
        let dim = self.dims as usize;
        let len = self.vectors.len();
        let mut sum = vec![0_f64; dim]; // change to f32 to avoid overflow
        for i in 0..len {
            let vec = self.vectors.get_vector(i);
            for j in 0..dim {
                sum[j] += f32::from(vec[j]) as f64;
            }
        }

        let collection: Vec<Scalar> = sum
            .iter()
            .map(|v| Scalar::from((*v / len as f64) as f32))
            .collect();
        collection
    }

    // r and l leave here for multiple pass extension
    fn _one_pass(
        &mut self,
        medoid: usize,
        alpha: f32,
        r: usize,
        l: usize,
        mut rng: impl Rng,
    ) -> Result<(), VamanaError> {
        let len = self.vectors.len();
        let mut ids = (0..len).collect::<Vec<_>>();
        ids.shuffle(&mut rng);

        for &id in ids.iter() {
            let query = self.vectors.get_vector(id);
            let state = self._greedy_search(medoid, query, 1, l)?;
            let neighbor_ids = self._robust_prune(id, state.visited, alpha, l)?;
            let neighbor_ids: HashSet<usize> = neighbor_ids.iter().cloned().collect();

            self._set_neighbors(id, &neighbor_ids);
            for &neighbor_id in neighbor_ids.iter() {
                let old_neighbors = self._get_neighbors(neighbor_id);
                let mut old_neighbors: HashSet<usize> = old_neighbors.iter().cloned().collect();
                old_neighbors.insert(id);
                if old_neighbors.len() > r {
                    // need robust prune
                    let new_neighbors = self._robust_prune(neighbor_id, old_neighbors, alpha, r)?;
                    let new_neighbors: HashSet<usize> = new_neighbors.iter().cloned().collect();
                    self._set_neighbors(neighbor_id, &new_neighbors);
                } else {
                    self._set_neighbors(neighbor_id, &old_neighbors);
                }
            }
        }

        Ok(())
    }

    fn _greedy_search(
        &self,
        start: usize,
        query: &[Scalar],
        k: usize,
        search_size: usize,
    ) -> Result<SearchState, VamanaError> {
        let mut state = SearchState::new(k, search_size);

        let dist = D::distance(query, self.vectors.get_vector(start));
        state.push(start, dist);
        while let Some(id) = state.pop() {
            // only pop id in the search list but not visited
            state.visit(id);

            let neighbor_ids = self._get_neighbors(id);
            for &neighbor_id in neighbor_ids {
                if state.is_visited(neighbor_id) {
                    continue;
                }

                let dist = D::distance(query, self.vectors.get_vector(neighbor_id));
                state.push(neighbor_id, dist); // push and retain closet l nodes
            }
        }

        Ok(state)
    }

    fn _robust_prune(
        &self,
        id: usize,
        mut visited: HashSet<usize>,
        alpha: f32,
        r: usize,
    ) -> Result<Vec<usize>, VamanaError> {
        // TODO: batch and concurrent
        visited.remove(&id); // in case visited has id itself
        let neighbor_ids = self._get_neighbors(id);
        visited.extend(neighbor_ids.iter());

        let mut heap: BinaryHeap<VertexWithDistance> = visited
            .iter()
            .map(|v| {
                let dist = D::distance(self.vectors.get_vector(id), self.vectors.get_vector(*v));
                VertexWithDistance {
                    id: *v,
                    distance: dist,
                }
            })
            .collect();

        let mut new_neighbor_ids: Vec<usize> = vec![];
        while !visited.is_empty() {
            let mut p = heap.pop().unwrap();
            while !visited.contains(&p.id) {
                p = heap.pop().unwrap();
            }

            new_neighbor_ids.push(p.id);
            if new_neighbor_ids.len() >= r {
                break;
            }
            let mut to_remove: HashSet<usize> = HashSet::new();
            for pv in visited.iter() {
                let dist_prime =
                    D::distance(self.vectors.get_vector(p.id), self.vectors.get_vector(*pv));
                let dist_query =
                    D::distance(self.vectors.get_vector(id), self.vectors.get_vector(*pv));
                if Scalar::from(alpha) * dist_prime <= dist_query {
                    to_remove.insert(*pv);
                }
            }
            for pv in to_remove.iter() {
                visited.remove(pv);
            }
        }
        Ok(new_neighbor_ids)
    }
}
