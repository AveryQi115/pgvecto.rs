use crate::prelude::*;
use std::simd::f32x4;
use std::simd::SimdFloat;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Distance {
    L2,
    Cosine,
    Dot,
}

impl Distance {
    #[inline(always)]
    pub fn distance(self, lhs: &[Scalar], rhs: &[Scalar]) -> Scalar {
        if is_vectorization_enabled() {
            match self {
                Distance::L2 => distance_squared_l2_vec(lhs, rhs),
                Distance::Cosine => distance_squared_cosine_vec(lhs, rhs) * (-1.0),
                Distance::Dot => distance_dot_vec(lhs, rhs) * (-1.0),
            }
        } else {
            match self {
                Distance::L2 => distance_squared_l2(lhs, rhs),
                Distance::Cosine => distance_squared_cosine(lhs, rhs) * (-1.0),
                Distance::Dot => distance_dot(lhs, rhs) * (-1.0),
            }
        }
    }
    #[inline(always)]
    pub fn kmeans_normalize(self, vector: &mut [Scalar]) {
        match self {
            Distance::L2 => (),
            Distance::Cosine | Distance::Dot => l2_normalize(vector),
        }
    }
    #[inline(always)]
    pub fn kmeans_distance(self, lhs: &[Scalar], rhs: &[Scalar]) -> Scalar {
        match self {
            Distance::L2 => distance_squared_l2(lhs, rhs),
            Distance::Cosine | Distance::Dot => distance_dot(lhs, rhs).acos(),
        }
    }
}

#[allow(unreachable_code)]
#[inline(always)]
fn is_vectorization_enabled() -> bool {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"),
                     any(target_feature = "sse",
                         target_feature = "sse2",
                         target_feature = "sse3",
                         target_feature = "ssse3",
                         target_feature = "sse4.1",
                         target_feature = "sse4.2",
                         target_feature = "sse4a")))] 
    {
        return true;
    }
    #[cfg(all(any(target_arch = "arm", target_arch = "aarch64"),
                            target_feature = "neon"))] 
    {
        return true;
    }
    false
}


#[inline(always)]
fn distance_squared_l2(lhs: &[Scalar], rhs: &[Scalar]) -> Scalar {
    if lhs.len() != rhs.len() {
        return Scalar::NAN;
    }
    let n = lhs.len();
    let mut result = Scalar::Z;
    for i in 0..n {
        let diff = lhs[i] - rhs[i];
        result += diff * diff;
    }
    result
}

#[inline(always)]
fn distance_squared_l2_vec(lhs: &[Scalar], rhs: &[Scalar]) -> Scalar {
    if lhs.len() != rhs.len() {
        return Scalar::NAN;
    }
    let lhs_f32: Vec<f32> = lhs.iter().map(|&item| f32::from(item)).collect();
    let lhs_f32_slice: &[f32] = &lhs_f32;

    let rhs_f32: Vec<f32> = rhs.iter().map(|&item| f32::from(item)).collect();
    let rhs_f32_slice: &[f32] = &rhs_f32;

    let (lhs_extra, lhs_chunks) = lhs_f32_slice.as_rchunks();
    let (rhs_extra, rhs_chunks) = rhs_f32_slice.as_rchunks();

    let mut sums = [0.0; 4];
    for ((x, y), d) in std::iter::zip(lhs_extra, rhs_extra).zip(&mut sums) {
        let diff = x - y;
        *d = diff * diff;
    }

    let mut sums = f32x4::from_array(sums);
    std::iter::zip(lhs_chunks, rhs_chunks).for_each(|(x, y)| {
        let diff = f32x4::from_array(*x) - f32x4::from_array(*y);
        sums += diff * diff;
    });

    Scalar(sums.reduce_sum())
}


#[inline(always)]
fn distance_squared_cosine(lhs: &[Scalar], rhs: &[Scalar]) -> Scalar {
    if lhs.len() != rhs.len() {
        return Scalar::NAN;
    }
    let n = lhs.len();
    let mut dot = Scalar::Z;
    let mut x2 = Scalar::Z;
    let mut y2 = Scalar::Z;
    for i in 0..n {
        dot += lhs[i] * rhs[i];
        x2 += lhs[i] * lhs[i];
        y2 += rhs[i] * rhs[i];
    }
    (dot * dot) / (x2 * y2)
}

#[inline(always)]
fn distance_squared_cosine_vec(lhs: &[Scalar], rhs: &[Scalar]) -> Scalar {
    if lhs.len() != rhs.len() {
        return Scalar::NAN;
    }

    let lhs_f32: Vec<f32> = lhs.iter().map(|&item| f32::from(item)).collect();
    let lhs_f32_slice: &[f32] = &lhs_f32;

    let rhs_f32: Vec<f32> = rhs.iter().map(|&item| f32::from(item)).collect();
    let rhs_f32_slice: &[f32] = &rhs_f32;

    let (lhs_extra, lhs_chunks) = lhs_f32_slice.as_rchunks();
    let (rhs_extra, rhs_chunks) = rhs_f32_slice.as_rchunks();

    let mut dot = [0.0; 4];
    let mut x2 = [0.0; 4];
    let mut y2 = [0.0; 4];
    for i in 0..lhs_extra.len() {
        let x = lhs_extra[i];
        let y = rhs_extra[i];
        dot[i] = x * y;
        x2[i] = x * x;
        y2[i] = y * y;
    }

    let mut dot = f32x4::from_array(dot);
    let mut x2 = f32x4::from_array(x2);
    let mut y2 = f32x4::from_array(y2);

    std::iter::zip(lhs_chunks, rhs_chunks).for_each(|(x, y)| {
        let x_vec = f32x4::from_array(*x);
        let y_vec = f32x4::from_array(*y);
        dot += x_vec * y_vec;
        x2 += x_vec * x_vec;
        y2 += y_vec * y_vec;
    });

    let dot = dot.reduce_sum();
    let x2 = x2.reduce_sum();
    let y2 = y2.reduce_sum();

    Scalar((dot * dot)/(x2 * y2))
}

#[inline(always)]
fn distance_dot(lhs: &[Scalar], rhs: &[Scalar]) -> Scalar {
    if lhs.len() != rhs.len() {
        return Scalar::NAN;
    }
    let n = lhs.len();
    let mut dot = Scalar::Z;
    for i in 0..n {
        dot += lhs[i] * rhs[i];
    }
    dot
}

#[inline(always)]
fn distance_dot_vec(lhs: &[Scalar], rhs: &[Scalar]) -> Scalar {
    if lhs.len() != rhs.len() {
        return Scalar::NAN;
    }
    let lhs_f32: Vec<f32> = lhs.iter().map(|&item| f32::from(item)).collect();
    let lhs_f32_slice: &[f32] = &lhs_f32;

    let rhs_f32: Vec<f32> = rhs.iter().map(|&item| f32::from(item)).collect();
    let rhs_f32_slice: &[f32] = &rhs_f32;

    let (lhs_extra, lhs_chunks) = lhs_f32_slice.as_rchunks();
    let (rhs_extra, rhs_chunks) = rhs_f32_slice.as_rchunks();

    let mut sums = [0.0; 4];
    for ((x, y), d) in std::iter::zip(lhs_extra, rhs_extra).zip(&mut sums) {
        *d = x * y;
    }

    let mut sums = f32x4::from_array(sums);
    std::iter::zip(lhs_chunks, rhs_chunks).for_each(|(x, y)| {
        sums += f32x4::from_array(*x) * f32x4::from_array(*y);
    });

    Scalar(sums.reduce_sum())
}

#[inline(always)]
fn length(vector: &[Scalar]) -> Scalar {
    let n = vector.len();
    let mut dot = Scalar::Z;
    for i in 0..n {
        dot += vector[i] * vector[i];
    }
    dot.sqrt()
}

#[inline(always)]
fn l2_normalize(vector: &mut [Scalar]) {
    let n = vector.len();
    let l = length(vector);
    for i in 0..n {
        vector[i] /= l;
    }
}

#[cfg(test)]
mod distance_tests {
    use rand::Rng;
    use super::*;

    #[test]
    fn test_distance_dot() {
        let mut rng = rand::thread_rng();
        if is_vectorization_enabled() {
            for _ in 0..100{
                let array_length = rng.gen_range(1..=10);
                let mut x = Vec::new();
                let mut y = Vec::new();

                for _ in 0..array_length {
                    let e1 = Scalar::from(rng.gen::<f32>());
                    let e2 = Scalar::from(rng.gen::<f32>());
                    x.push(e1);
                    y.push(e2);
                }

                let x: &[Scalar] = &x;
                let y: &[Scalar] = &y;
                assert!((distance_dot(x,y)-distance_dot_vec(x,y)).0.abs() <= 1e-5);
            }
        }
    }

    #[test]
    fn test_distance_squared_cosine() {
        let mut rng = rand::thread_rng();
        if is_vectorization_enabled() {
            for _ in 0..100{
                let array_length = rng.gen_range(1..=10);
                let mut x = Vec::new();
                let mut y = Vec::new();

                for _ in 0..array_length {
                    let e1 = Scalar::from(rng.gen::<f32>());
                    let e2 = Scalar::from(rng.gen::<f32>());
                    x.push(e1);
                    y.push(e2);
                }

                let x: &[Scalar] = &x;
                let y: &[Scalar] = &y;
                assert!((distance_squared_cosine(x,y)-distance_squared_cosine_vec(x,y)).0.abs() <= 1e-5);
            }
        }
    }

    #[test]
    fn test_distance_squared_l2() {
        let mut rng = rand::thread_rng();
        if is_vectorization_enabled() {
            for _ in 0..100{
                let array_length = rng.gen_range(1..=10);
                let mut x = Vec::new();
                let mut y = Vec::new();

                for _ in 0..array_length {
                    let e1 = Scalar::from(rng.gen::<f32>());
                    let e2 = Scalar::from(rng.gen::<f32>());
                    x.push(e1);
                    y.push(e2);
                }

                let x: &[Scalar] = &x;
                let y: &[Scalar] = &y;
                assert!((distance_squared_l2(x,y)-distance_squared_l2_vec(x,y)).0.abs() <= 1e-5);
            }
        }
    }
}
