//! Compiled model artifact: assembles outputs from all passes into
//! a deployable package.

use crate::passes::base::CompilerOutput;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// A compiled ExQ model artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledArtifact {
    pub model_id: String,
    /// (layer, expert) → precision name.
    pub quant_map: HashMap<String, String>,
    /// (layer, expert) → memory offset.
    pub layout_map: HashMap<String, u64>,
    /// Layer → specialization kind.
    pub specialization_map: HashMap<String, String>,
    /// Total prefetch entries emitted.
    pub prefetch_entry_count: usize,
    /// Static prefetch coverage ratio.
    pub prefetch_coverage: f64,
    /// Estimated total memory in bytes.
    pub estimated_memory_bytes: u64,
}

impl CompiledArtifact {
    /// Build from compiler output.
    pub fn from_compiler_output(model_id: &str, output: &CompilerOutput, coverage: f64) -> Self {
        let quant_map: HashMap<String, String> = output
            .quant
            .assignments
            .iter()
            .map(|(&(l, e), p)| (format!("{l}:{e}"), format!("{p:?}")))
            .collect();

        let layout_map: HashMap<String, u64> = output
            .layout
            .placements
            .iter()
            .map(|(&(l, e), &off)| (format!("{l}:{e}"), off))
            .collect();

        let specialization_map: HashMap<String, String> = output
            .specialization
            .layer_decisions
            .iter()
            .map(|(&l, d)| {
                let kind = match d {
                    crate::passes::specialization::LayerSpecialization::General => "General",
                    crate::passes::specialization::LayerSpecialization::MultiPath { .. } => {
                        "MultiPath"
                    }
                    crate::passes::specialization::LayerSpecialization::FastPath { .. } => {
                        "FastPath"
                    }
                };
                (l.to_string(), kind.to_string())
            })
            .collect();

        Self {
            model_id: model_id.to_string(),
            quant_map,
            layout_map,
            specialization_map,
            prefetch_entry_count: output.prefetch.entries.len(),
            prefetch_coverage: coverage,
            estimated_memory_bytes: output.quant.estimated_memory_bytes,
        }
    }

    /// Save artifact metadata to JSON.
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let f = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(f, self)?;
        Ok(())
    }

    /// Load artifact metadata from JSON.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let f = std::fs::File::open(path)?;
        let artifact: Self = serde_json::from_reader(f)?;
        Ok(artifact)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::routing_graph::{RoutingGraph, RoutingGraphEdge, RoutingGraphNode};
    use crate::passes::base::run_pipeline;

    #[test]
    fn test_artifact_from_pipeline() {
        let mut g = RoutingGraph::new("artifact-test".into());
        for layer in 0..2 {
            for expert in 0..4 {
                g.add_node(RoutingGraphNode {
                    layer,
                    expert,
                    activation_freq: [0.35, 0.30, 0.20, 0.15][expert],
                    weight_size_bytes: 1_000_000,
                    avg_arithmetic_intensity: 100.0,
                    routing_entropy: 1.3,
                });
            }
        }
        g.add_edge(RoutingGraphEdge {
            src_layer: 0,
            src_expert: 0,
            dst_layer: 1,
            dst_expert: 0,
            conditional_prob: 0.80,
        });

        let output = run_pipeline(&g);
        let artifact = CompiledArtifact::from_compiler_output("test", &output, 0.85);

        assert_eq!(artifact.model_id, "test");
        assert!(!artifact.quant_map.is_empty());
        assert!(!artifact.layout_map.is_empty());
    }

    #[test]
    fn test_artifact_json_roundtrip() {
        let artifact = CompiledArtifact {
            model_id: "test-model".into(),
            quant_map: HashMap::from([("0:0".into(), "BF16".into())]),
            layout_map: HashMap::from([("0:0".into(), 0)]),
            specialization_map: HashMap::from([("0".into(), "General".into())]),
            prefetch_entry_count: 5,
            prefetch_coverage: 0.85,
            estimated_memory_bytes: 10_000_000,
        };

        let tmp = std::env::temp_dir().join("exq_test_artifact.json");
        artifact.save(&tmp).unwrap();
        let loaded = CompiledArtifact::load(&tmp).unwrap();
        assert_eq!(loaded.model_id, "test-model");
        assert_eq!(loaded.prefetch_entry_count, 5);
        std::fs::remove_file(&tmp).ok();
    }
}
