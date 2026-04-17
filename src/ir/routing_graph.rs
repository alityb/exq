//! Core routing graph data structures: nodes, edges, and the graph itself.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Pure-Rust types
// ---------------------------------------------------------------------------

/// A node in the routing graph: one (layer, expert) pair.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoutingGraphNode {
    pub layer: usize,
    pub expert: usize,
    /// P(this expert activates) over the calibration corpus.
    pub activation_freq: f64,
    /// Size of this expert's parameters in bytes.
    pub weight_size_bytes: u64,
    /// FLOP/byte ratio when this expert runs.
    pub avg_arithmetic_intensity: f64,
    /// Entropy of the router at this layer (shared across all experts in layer).
    pub routing_entropy: f64,
}

impl RoutingGraphNode {
    pub fn key(&self) -> (usize, usize) {
        (self.layer, self.expert)
    }
}

/// An edge: cross-layer conditional co-activation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoutingGraphEdge {
    pub src_layer: usize,
    pub src_expert: usize,
    pub dst_layer: usize,
    pub dst_expert: usize,
    /// P(dst activates | src activated).
    pub conditional_prob: f64,
}

impl RoutingGraphEdge {
    pub fn src_key(&self) -> (usize, usize) {
        (self.src_layer, self.src_expert)
    }
    pub fn dst_key(&self) -> (usize, usize) {
        (self.dst_layer, self.dst_expert)
    }
}

/// The central IR consumed by all compiler passes.
///
/// `nodes` uses tuple keys `(layer, expert)` which can't be JSON map keys,
/// so we serialize via a Vec of nodes and rebuild the HashMap on deser.
#[derive(Debug, Clone)]
pub struct RoutingGraph {
    pub model_id: String,
    pub nodes: HashMap<(usize, usize), RoutingGraphNode>,
    pub edges: Vec<RoutingGraphEdge>,
    /// Adjacency list: src_key -> list of edge indices
    pub adj_out: HashMap<(usize, usize), Vec<usize>>,
    /// Reverse adjacency: dst_key -> list of edge indices
    pub adj_in: HashMap<(usize, usize), Vec<usize>>,
}

/// Serializable form of a RoutingGraph (nodes as Vec, no adjacency).
#[derive(Serialize, Deserialize)]
struct RoutingGraphSerde {
    model_id: String,
    nodes: Vec<RoutingGraphNode>,
    edges: Vec<RoutingGraphEdge>,
}

impl Serialize for RoutingGraph {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let serde_form = RoutingGraphSerde {
            model_id: self.model_id.clone(),
            nodes: self.nodes.values().cloned().collect(),
            edges: self.edges.clone(),
        };
        serde_form.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RoutingGraph {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let serde_form = RoutingGraphSerde::deserialize(deserializer)?;
        let mut g = RoutingGraph::new(serde_form.model_id);
        for node in serde_form.nodes {
            g.nodes.insert(node.key(), node);
        }
        for edge in serde_form.edges {
            g.add_edge(edge);
        }
        Ok(g)
    }
}

impl RoutingGraph {
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            nodes: HashMap::new(),
            edges: Vec::new(),
            adj_out: HashMap::new(),
            adj_in: HashMap::new(),
        }
    }

    /// Insert a node; overwrites if key already exists.
    pub fn add_node(&mut self, node: RoutingGraphNode) {
        self.nodes.insert(node.key(), node);
    }

    /// Insert an edge; updates adjacency lists.
    pub fn add_edge(&mut self, edge: RoutingGraphEdge) {
        let idx = self.edges.len();
        self.adj_out.entry(edge.src_key()).or_default().push(idx);
        self.adj_in.entry(edge.dst_key()).or_default().push(idx);
        self.edges.push(edge);
    }

    /// Rebuild adjacency lists (needed after deserialization).
    pub fn rebuild_adjacency(&mut self) {
        self.adj_out.clear();
        self.adj_in.clear();
        for (idx, e) in self.edges.iter().enumerate() {
            self.adj_out.entry(e.src_key()).or_default().push(idx);
            self.adj_in.entry(e.dst_key()).or_default().push(idx);
        }
    }

    /// Sorted list of unique layer indices.
    pub fn layer_indices(&self) -> Vec<usize> {
        let mut layers: Vec<usize> = self.nodes.keys().map(|k| k.0).collect();
        layers.sort();
        layers.dedup();
        layers
    }

    /// All nodes in a given layer.
    pub fn nodes_in_layer(&self, layer: usize) -> Vec<&RoutingGraphNode> {
        self.nodes.values().filter(|n| n.layer == layer).collect()
    }

    /// Hot experts: activation_freq >= threshold.
    pub fn hot_experts(&self, threshold: f64) -> Vec<&RoutingGraphNode> {
        self.nodes
            .values()
            .filter(|n| n.activation_freq >= threshold)
            .collect()
    }

    /// Cold experts: activation_freq < threshold.
    pub fn cold_experts(&self, threshold: f64) -> Vec<&RoutingGraphNode> {
        self.nodes
            .values()
            .filter(|n| n.activation_freq < threshold)
            .collect()
    }

    /// Edges with conditional_prob >= min_prob.
    pub fn high_prob_edges(&self, min_prob: f64) -> Vec<&RoutingGraphEdge> {
        self.edges
            .iter()
            .filter(|e| e.conditional_prob >= min_prob)
            .collect()
    }

    /// Outgoing edges from a node.
    pub fn outgoing_edges(&self, key: (usize, usize)) -> Vec<&RoutingGraphEdge> {
        self.adj_out
            .get(&key)
            .map(|idxs| idxs.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    /// Incoming edges to a node.
    pub fn incoming_edges(&self, key: (usize, usize)) -> Vec<&RoutingGraphEdge> {
        self.adj_in
            .get(&key)
            .map(|idxs| idxs.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    /// Total number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string (adjacency rebuilt by Deserialize impl).
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }
}

// ---------------------------------------------------------------------------
// PyO3 wrappers
// ---------------------------------------------------------------------------

#[pyclass(name = "RoutingGraphNode")]
#[derive(Clone)]
pub struct PyRoutingGraphNode {
    pub inner: RoutingGraphNode,
}

#[pymethods]
impl PyRoutingGraphNode {
    #[new]
    #[pyo3(signature = (layer, expert, activation_freq=0.0, weight_size_bytes=0, avg_arithmetic_intensity=0.0, routing_entropy=0.0))]
    fn new(
        layer: usize,
        expert: usize,
        activation_freq: f64,
        weight_size_bytes: u64,
        avg_arithmetic_intensity: f64,
        routing_entropy: f64,
    ) -> Self {
        Self {
            inner: RoutingGraphNode {
                layer,
                expert,
                activation_freq,
                weight_size_bytes,
                avg_arithmetic_intensity,
                routing_entropy,
            },
        }
    }

    #[getter]
    fn layer(&self) -> usize {
        self.inner.layer
    }
    #[getter]
    fn expert(&self) -> usize {
        self.inner.expert
    }
    #[getter]
    fn activation_freq(&self) -> f64 {
        self.inner.activation_freq
    }
    #[getter]
    fn weight_size_bytes(&self) -> u64 {
        self.inner.weight_size_bytes
    }
    #[getter]
    fn routing_entropy(&self) -> f64 {
        self.inner.routing_entropy
    }

    fn __repr__(&self) -> String {
        format!(
            "RoutingGraphNode(layer={}, expert={}, freq={:.4})",
            self.inner.layer, self.inner.expert, self.inner.activation_freq
        )
    }
}

#[pyclass(name = "RoutingGraphEdge")]
#[derive(Clone)]
pub struct PyRoutingGraphEdge {
    pub inner: RoutingGraphEdge,
}

#[pymethods]
impl PyRoutingGraphEdge {
    #[new]
    fn new(
        src_layer: usize,
        src_expert: usize,
        dst_layer: usize,
        dst_expert: usize,
        conditional_prob: f64,
    ) -> Self {
        Self {
            inner: RoutingGraphEdge {
                src_layer,
                src_expert,
                dst_layer,
                dst_expert,
                conditional_prob,
            },
        }
    }

    #[getter]
    fn src_layer(&self) -> usize {
        self.inner.src_layer
    }
    #[getter]
    fn src_expert(&self) -> usize {
        self.inner.src_expert
    }
    #[getter]
    fn dst_layer(&self) -> usize {
        self.inner.dst_layer
    }
    #[getter]
    fn dst_expert(&self) -> usize {
        self.inner.dst_expert
    }
    #[getter]
    fn conditional_prob(&self) -> f64 {
        self.inner.conditional_prob
    }

    fn __repr__(&self) -> String {
        format!(
            "RoutingGraphEdge(L{}:E{} -> L{}:E{}, p={:.4})",
            self.inner.src_layer,
            self.inner.src_expert,
            self.inner.dst_layer,
            self.inner.dst_expert,
            self.inner.conditional_prob,
        )
    }
}

#[pyclass(name = "RoutingGraph")]
#[derive(Clone)]
pub struct PyRoutingGraph {
    pub inner: RoutingGraph,
}

#[pymethods]
impl PyRoutingGraph {
    #[new]
    fn new(model_id: String) -> Self {
        Self {
            inner: RoutingGraph::new(model_id),
        }
    }

    #[getter]
    fn model_id(&self) -> &str {
        &self.inner.model_id
    }
    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }
    #[getter]
    fn n_edges(&self) -> usize {
        self.inner.n_edges()
    }

    fn add_node(&mut self, node: &PyRoutingGraphNode) {
        self.inner.add_node(node.inner.clone());
    }

    fn add_edge(&mut self, edge: &PyRoutingGraphEdge) {
        self.inner.add_edge(edge.inner.clone());
    }

    fn layer_indices(&self) -> Vec<usize> {
        self.inner.layer_indices()
    }

    /// Get hot experts as list of (layer, expert, freq) tuples.
    #[pyo3(signature = (threshold=0.10))]
    fn hot_experts(&self, threshold: f64) -> Vec<(usize, usize, f64)> {
        self.inner
            .hot_experts(threshold)
            .into_iter()
            .map(|n| (n.layer, n.expert, n.activation_freq))
            .collect()
    }

    /// Get cold experts as list of (layer, expert, freq) tuples.
    #[pyo3(signature = (threshold=0.01))]
    fn cold_experts(&self, threshold: f64) -> Vec<(usize, usize, f64)> {
        self.inner
            .cold_experts(threshold)
            .into_iter()
            .map(|n| (n.layer, n.expert, n.activation_freq))
            .collect()
    }

    /// Get high-probability edges as list of (src_l, src_e, dst_l, dst_e, prob) tuples.
    #[pyo3(signature = (min_prob=0.60))]
    fn high_prob_edges(&self, min_prob: f64) -> Vec<(usize, usize, usize, usize, f64)> {
        self.inner
            .high_prob_edges(min_prob)
            .into_iter()
            .map(|e| {
                (
                    e.src_layer,
                    e.src_expert,
                    e.dst_layer,
                    e.dst_expert,
                    e.conditional_prob,
                )
            })
            .collect()
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn from_json(json_str: &str) -> PyResult<Self> {
        RoutingGraph::from_json(json_str)
            .map(|inner| Self { inner })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "RoutingGraph(model='{}', nodes={}, edges={})",
            self.inner.model_id,
            self.inner.n_nodes(),
            self.inner.n_edges(),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_graph() -> RoutingGraph {
        let mut g = RoutingGraph::new("test".into());
        // 2 layers, 4 experts each
        for layer in 0..2 {
            for expert in 0..4 {
                let freq = if expert < 2 { 0.3 } else { 0.05 };
                g.add_node(RoutingGraphNode {
                    layer,
                    expert,
                    activation_freq: freq,
                    weight_size_bytes: 1_000_000,
                    avg_arithmetic_intensity: 100.0,
                    routing_entropy: 1.2,
                });
            }
        }
        // Edges L0 -> L1
        g.add_edge(RoutingGraphEdge {
            src_layer: 0,
            src_expert: 0,
            dst_layer: 1,
            dst_expert: 0,
            conditional_prob: 0.8,
        });
        g.add_edge(RoutingGraphEdge {
            src_layer: 0,
            src_expert: 0,
            dst_layer: 1,
            dst_expert: 1,
            conditional_prob: 0.2,
        });
        g.add_edge(RoutingGraphEdge {
            src_layer: 0,
            src_expert: 1,
            dst_layer: 1,
            dst_expert: 2,
            conditional_prob: 0.5,
        });
        g
    }

    #[test]
    fn test_node_count() {
        let g = make_test_graph();
        assert_eq!(g.n_nodes(), 8);
    }

    #[test]
    fn test_edge_count() {
        let g = make_test_graph();
        assert_eq!(g.n_edges(), 3);
    }

    #[test]
    fn test_hot_experts() {
        let g = make_test_graph();
        let hot = g.hot_experts(0.10);
        assert_eq!(hot.len(), 4); // experts 0,1 in both layers
        for n in &hot {
            assert!(n.activation_freq >= 0.10);
        }
    }

    #[test]
    fn test_cold_experts() {
        let g = make_test_graph();
        let cold = g.cold_experts(0.10);
        assert_eq!(cold.len(), 4); // experts 2,3 in both layers
    }

    #[test]
    fn test_high_prob_edges() {
        let g = make_test_graph();
        let high = g.high_prob_edges(0.60);
        assert_eq!(high.len(), 1);
        assert_eq!(high[0].src_expert, 0);
        assert_eq!(high[0].dst_expert, 0);
    }

    #[test]
    fn test_outgoing_edges() {
        let g = make_test_graph();
        let out = g.outgoing_edges((0, 0));
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_layer_indices() {
        let g = make_test_graph();
        assert_eq!(g.layer_indices(), vec![0, 1]);
    }

    #[test]
    fn test_json_roundtrip() {
        let g = make_test_graph();
        let json = g.to_json().unwrap();
        let g2 = RoutingGraph::from_json(&json).unwrap();
        assert_eq!(g2.n_nodes(), g.n_nodes());
        assert_eq!(g2.n_edges(), g.n_edges());
        // Adjacency rebuilt
        assert_eq!(g2.outgoing_edges((0, 0)).len(), 2);
    }
}
