//! Routing profile schema: structs for profiler output + serde + PyO3.
//!
//! Mirrors the JSON schema from AGENTS.md §5.2. Every field is
//! (de)serializable via serde_json and accessible from Python via PyO3.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Pure-Rust types (used internally + in tests)
// ---------------------------------------------------------------------------

/// Per-expert statistics within a single MoE layer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExpertStats {
    pub expert_id: usize,
    pub activation_count: u64,
    pub activation_freq: f64,
    pub avg_input_l2_norm: f64,
}

impl ExpertStats {
    pub fn new(expert_id: usize) -> Self {
        Self {
            expert_id,
            activation_count: 0,
            activation_freq: 0.0,
            avg_input_l2_norm: 0.0,
        }
    }
}

/// Routing profile for a single MoE layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerProfile {
    pub layer_idx: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub expert_stats: Vec<ExpertStats>,
    pub routing_entropy: f64,
    /// co_activation_next_layer[src_expert_id] -> {dst_expert_id -> P(dst|src)}
    #[serde(default)]
    pub co_activation_next_layer: HashMap<usize, HashMap<usize, f64>>,
}

impl LayerProfile {
    pub fn new(layer_idx: usize, n_experts: usize, top_k: usize) -> Self {
        let expert_stats = (0..n_experts).map(ExpertStats::new).collect();
        Self {
            layer_idx,
            n_experts,
            top_k,
            expert_stats,
            routing_entropy: 0.0,
            co_activation_next_layer: HashMap::new(),
        }
    }

    /// Normalize raw activation counts → frequencies summing to 1.0.
    pub fn normalize_frequencies(&mut self) {
        let total: u64 = self.expert_stats.iter().map(|s| s.activation_count).sum();
        if total == 0 {
            return;
        }
        for s in &mut self.expert_stats {
            s.activation_freq = s.activation_count as f64 / total as f64;
        }
    }

    /// Compute Shannon entropy of the routing distribution (nats).
    pub fn compute_entropy(&mut self) -> f64 {
        let mut h = 0.0f64;
        for s in &self.expert_stats {
            let p = s.activation_freq;
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        self.routing_entropy = h;
        h
    }

    /// Normalize co-activation counts → conditional probabilities.
    pub fn normalize_co_activations(&mut self) {
        for (_src, dsts) in self.co_activation_next_layer.iter_mut() {
            let total: f64 = dsts.values().sum();
            if total > 0.0 {
                for p in dsts.values_mut() {
                    *p /= total;
                }
            }
        }
    }

    /// Total number of activations across all experts.
    pub fn total_activations(&self) -> u64 {
        self.expert_stats.iter().map(|s| s.activation_count).sum()
    }
}

/// Full routing profile across all MoE layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingProfile {
    pub model_id: String,
    pub calibration_samples: usize,
    #[serde(default)]
    pub calibration_tokens: usize,
    /// layer_idx → LayerProfile
    pub layers: HashMap<usize, LayerProfile>,
}

impl RoutingProfile {
    pub fn new(model_id: String, calibration_samples: usize) -> Self {
        Self {
            model_id,
            calibration_samples,
            calibration_tokens: 0,
            layers: HashMap::new(),
        }
    }

    /// Sorted list of layer indices.
    pub fn moe_layer_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self.layers.keys().copied().collect();
        indices.sort();
        indices
    }

    /// Validate integrity; returns list of warning messages.
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        for (&layer_idx, lp) in &self.layers {
            let freq_sum: f64 = lp.expert_stats.iter().map(|s| s.activation_freq).sum();
            if (freq_sum - 1.0).abs() > 1e-4 {
                warnings.push(format!(
                    "Layer {layer_idx}: frequencies sum to {freq_sum:.6}, expected 1.0"
                ));
            }
            for (&src_e, dsts) in &lp.co_activation_next_layer {
                let p_sum: f64 = dsts.values().sum();
                if (p_sum - 1.0).abs() > 1e-3 && p_sum > 0.0 {
                    warnings.push(format!(
                        "Layer {layer_idx}, expert {src_e}: co-activation probs sum to {p_sum:.4}"
                    ));
                }
            }
        }
        warnings
    }

    /// Serialize to JSON file.
    pub fn save(&self, path: &Path) -> Result<(), ProfileError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let f = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(f, self)?;
        Ok(())
    }

    /// Deserialize from JSON file.
    pub fn load(path: &Path) -> Result<Self, ProfileError> {
        let f = std::fs::File::open(path)?;
        let profile: Self = serde_json::from_reader(f)?;
        Ok(profile)
    }

    /// Deserialize from JSON string.
    pub fn from_json(json_str: &str) -> Result<Self, ProfileError> {
        let profile: Self = serde_json::from_str(json_str)?;
        Ok(profile)
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, ProfileError> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Validation error: {0}")]
    Validation(String),
}

// ---------------------------------------------------------------------------
// PyO3 wrappers
// ---------------------------------------------------------------------------

/// Python-exposed ExpertStats.
#[pyclass(name = "ExpertStats")]
#[derive(Clone)]
pub struct PyExpertStats {
    pub inner: ExpertStats,
}

#[pymethods]
impl PyExpertStats {
    #[new]
    fn new(expert_id: usize) -> Self {
        Self {
            inner: ExpertStats::new(expert_id),
        }
    }

    #[getter]
    fn expert_id(&self) -> usize {
        self.inner.expert_id
    }
    #[getter]
    fn activation_count(&self) -> u64 {
        self.inner.activation_count
    }
    #[setter]
    fn set_activation_count(&mut self, v: u64) {
        self.inner.activation_count = v;
    }
    #[getter]
    fn activation_freq(&self) -> f64 {
        self.inner.activation_freq
    }
    #[getter]
    fn avg_input_l2_norm(&self) -> f64 {
        self.inner.avg_input_l2_norm
    }

    fn __repr__(&self) -> String {
        format!(
            "ExpertStats(id={}, count={}, freq={:.4})",
            self.inner.expert_id, self.inner.activation_count, self.inner.activation_freq
        )
    }
}

/// Python-exposed LayerProfile.
#[pyclass(name = "LayerProfile")]
#[derive(Clone)]
pub struct PyLayerProfile {
    pub inner: LayerProfile,
}

#[pymethods]
impl PyLayerProfile {
    #[new]
    fn new(layer_idx: usize, n_experts: usize, top_k: usize) -> Self {
        Self {
            inner: LayerProfile::new(layer_idx, n_experts, top_k),
        }
    }

    #[getter]
    fn layer_idx(&self) -> usize {
        self.inner.layer_idx
    }
    #[getter]
    fn n_experts(&self) -> usize {
        self.inner.n_experts
    }
    #[getter]
    fn top_k(&self) -> usize {
        self.inner.top_k
    }
    #[getter]
    fn routing_entropy(&self) -> f64 {
        self.inner.routing_entropy
    }

    /// Get expert activation frequencies as a list.
    fn get_activation_freqs(&self) -> Vec<f64> {
        self.inner
            .expert_stats
            .iter()
            .map(|s| s.activation_freq)
            .collect()
    }

    /// Get expert activation counts as a list.
    fn get_activation_counts(&self) -> Vec<u64> {
        self.inner
            .expert_stats
            .iter()
            .map(|s| s.activation_count)
            .collect()
    }

    /// Set activation count for a single expert.
    fn set_expert_count(&mut self, expert_id: usize, count: u64) -> PyResult<()> {
        if expert_id >= self.inner.n_experts {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "expert_id {expert_id} >= n_experts {}",
                self.inner.n_experts
            )));
        }
        self.inner.expert_stats[expert_id].activation_count = count;
        Ok(())
    }

    /// Increment activation count for a single expert.
    fn increment_expert(&mut self, expert_id: usize) -> PyResult<()> {
        if expert_id >= self.inner.n_experts {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "expert_id {expert_id} >= n_experts {}",
                self.inner.n_experts
            )));
        }
        self.inner.expert_stats[expert_id].activation_count += 1;
        Ok(())
    }

    /// Add a co-activation observation: src_expert at this layer → dst_expert at next layer.
    fn add_co_activation(&mut self, src_expert: usize, dst_expert: usize) {
        *self
            .inner
            .co_activation_next_layer
            .entry(src_expert)
            .or_default()
            .entry(dst_expert)
            .or_insert(0.0) += 1.0;
    }

    /// Normalize counts → frequencies + compute entropy + normalize co-activations.
    fn finalize(&mut self) {
        self.inner.normalize_frequencies();
        self.inner.compute_entropy();
        self.inner.normalize_co_activations();
    }

    fn __repr__(&self) -> String {
        format!(
            "LayerProfile(layer={}, n_experts={}, top_k={}, entropy={:.4})",
            self.inner.layer_idx,
            self.inner.n_experts,
            self.inner.top_k,
            self.inner.routing_entropy,
        )
    }
}

/// Python-exposed RoutingProfile.
#[pyclass(name = "RoutingProfile")]
#[derive(Clone)]
pub struct PyRoutingProfile {
    pub inner: RoutingProfile,
}

#[pymethods]
impl PyRoutingProfile {
    #[new]
    fn new(model_id: String, calibration_samples: usize) -> Self {
        Self {
            inner: RoutingProfile::new(model_id, calibration_samples),
        }
    }

    #[getter]
    fn model_id(&self) -> &str {
        &self.inner.model_id
    }
    #[getter]
    fn calibration_samples(&self) -> usize {
        self.inner.calibration_samples
    }
    #[getter]
    fn calibration_tokens(&self) -> usize {
        self.inner.calibration_tokens
    }
    #[setter]
    fn set_calibration_tokens(&mut self, v: usize) {
        self.inner.calibration_tokens = v;
    }
    #[getter]
    fn n_layers(&self) -> usize {
        self.inner.layers.len()
    }

    /// Add a new layer profile.
    fn add_layer(&mut self, layer: &PyLayerProfile) {
        self.inner
            .layers
            .insert(layer.inner.layer_idx, layer.inner.clone());
    }

    /// Get layer profile by index.
    fn get_layer(&self, layer_idx: usize) -> PyResult<PyLayerProfile> {
        self.inner
            .layers
            .get(&layer_idx)
            .map(|lp| PyLayerProfile { inner: lp.clone() })
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("layer {layer_idx} not found"))
            })
    }

    /// Sorted list of MoE layer indices.
    fn moe_layer_indices(&self) -> Vec<usize> {
        self.inner.moe_layer_indices()
    }

    /// Validate and return warnings.
    fn validate(&self) -> Vec<String> {
        self.inner.validate()
    }

    /// Save to JSON file.
    fn save(&self, path: String) -> PyResult<()> {
        self.inner
            .save(Path::new(&path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Load from JSON file.
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        RoutingProfile::load(Path::new(&path))
            .map(|inner| Self { inner })
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Load from JSON string.
    #[staticmethod]
    fn from_json(json_str: &str) -> PyResult<Self> {
        RoutingProfile::from_json(json_str)
            .map(|inner| Self { inner })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Serialize to JSON string.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "RoutingProfile(model='{}', samples={}, layers={})",
            self.inner.model_id,
            self.inner.calibration_samples,
            self.inner.layers.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Rust-side tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_layer(n_experts: usize, top_k: usize) -> LayerProfile {
        let mut lp = LayerProfile::new(0, n_experts, top_k);
        // Simulate some activations: first few experts are hot
        for (i, s) in lp.expert_stats.iter_mut().enumerate() {
            s.activation_count = ((n_experts - i) * 10) as u64;
        }
        lp.normalize_frequencies();
        lp.compute_entropy();
        lp
    }

    #[test]
    fn test_frequencies_sum_to_one() {
        let lp = make_test_layer(8, 2);
        let sum: f64 = lp.expert_stats.iter().map(|s| s.activation_freq).sum();
        assert!((sum - 1.0).abs() < 1e-10, "freq sum = {sum}");
    }

    #[test]
    fn test_entropy_non_negative() {
        let lp = make_test_layer(8, 2);
        assert!(lp.routing_entropy >= 0.0);
    }

    #[test]
    fn test_uniform_has_max_entropy() {
        let mut lp = LayerProfile::new(0, 4, 1);
        for s in &mut lp.expert_stats {
            s.activation_count = 100;
        }
        lp.normalize_frequencies();
        let h = lp.compute_entropy();
        let h_max = (4.0f64).ln();
        assert!(
            (h - h_max).abs() < 1e-10,
            "uniform entropy should be ln(4)={h_max}, got {h}"
        );
    }

    #[test]
    fn test_single_expert_zero_entropy() {
        let mut lp = LayerProfile::new(0, 4, 1);
        lp.expert_stats[0].activation_count = 100;
        lp.normalize_frequencies();
        let h = lp.compute_entropy();
        assert!(
            (h - 0.0).abs() < 1e-10,
            "single expert should have 0 entropy, got {h}"
        );
    }

    #[test]
    fn test_co_activation_normalization() {
        let mut lp = LayerProfile::new(0, 4, 1);
        lp.co_activation_next_layer
            .entry(0)
            .or_default()
            .insert(1, 30.0);
        lp.co_activation_next_layer
            .entry(0)
            .or_default()
            .insert(2, 70.0);
        lp.normalize_co_activations();
        let dsts = &lp.co_activation_next_layer[&0];
        assert!((dsts[&1] - 0.3).abs() < 1e-10);
        assert!((dsts[&2] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_json_roundtrip() {
        let mut profile = RoutingProfile::new("test-model".into(), 100);
        let lp = make_test_layer(4, 2);
        profile.layers.insert(0, lp);
        let json = profile.to_json().unwrap();
        let restored = RoutingProfile::from_json(&json).unwrap();
        assert_eq!(restored.model_id, "test-model");
        assert_eq!(restored.layers.len(), 1);
        let rl = &restored.layers[&0];
        for (orig, rest) in profile.layers[&0]
            .expert_stats
            .iter()
            .zip(rl.expert_stats.iter())
        {
            assert_eq!(orig.expert_id, rest.expert_id);
            assert_eq!(orig.activation_count, rest.activation_count);
            assert!((orig.activation_freq - rest.activation_freq).abs() < 1e-10);
        }
    }

    #[test]
    fn test_validate_good_profile() {
        let mut profile = RoutingProfile::new("test".into(), 50);
        let lp = make_test_layer(8, 2);
        profile.layers.insert(0, lp);
        let warnings = profile.validate();
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    }

    #[test]
    fn test_validate_bad_freqs() {
        let mut profile = RoutingProfile::new("test".into(), 50);
        let mut lp = LayerProfile::new(0, 4, 1);
        lp.expert_stats[0].activation_freq = 0.5; // doesn't sum to 1
        profile.layers.insert(0, lp);
        let warnings = profile.validate();
        assert!(!warnings.is_empty());
    }

    #[test]
    fn test_file_roundtrip() {
        let mut profile = RoutingProfile::new("file-test".into(), 200);
        let lp = make_test_layer(8, 2);
        profile.layers.insert(0, lp);

        let tmp = std::env::temp_dir().join("exq_test_profile.json");
        profile.save(&tmp).unwrap();
        let loaded = RoutingProfile::load(&tmp).unwrap();
        assert_eq!(loaded.model_id, "file-test");
        assert_eq!(loaded.calibration_samples, 200);
        std::fs::remove_file(&tmp).ok();
    }
}
