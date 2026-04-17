//! Kernel generation stubs.
//!
//! In the full implementation, this module emits Triton/CUDA kernels
//! with statically embedded prefetch instructions. For v0.1, it generates
//! a human-readable pseudo-kernel representation for inspection.

use crate::passes::prefetch_emitter::{PrefetchEntry, PrefetchPriority, PrefetchSchedule};

/// A pseudo-kernel representation for a single (layer, expert).
#[derive(Debug, Clone)]
pub struct PseudoKernel {
    pub layer: usize,
    pub expert: usize,
    pub prefetch_instructions: Vec<PrefetchInstruction>,
}

/// A single prefetch instruction in the pseudo-kernel.
#[derive(Debug, Clone)]
pub struct PrefetchInstruction {
    pub target_layer: usize,
    pub target_expert: usize,
    pub priority: PrefetchPriority,
    pub size_bytes: u64,
    pub comment: String,
}

/// Generate pseudo-kernels from a prefetch schedule.
pub fn generate_pseudo_kernels(schedule: &PrefetchSchedule) -> Vec<PseudoKernel> {
    use std::collections::HashMap;

    // Group entries by (src_layer, src_expert)
    let mut grouped: HashMap<(usize, usize), Vec<&PrefetchEntry>> = HashMap::new();
    for entry in &schedule.entries {
        grouped
            .entry((entry.src_layer, entry.src_expert))
            .or_default()
            .push(entry);
    }

    let mut kernels: Vec<PseudoKernel> = grouped
        .into_iter()
        .map(|((layer, expert), entries)| {
            let mut instructions: Vec<PrefetchInstruction> = entries
                .into_iter()
                .map(|e| {
                    let priority_str = match e.priority {
                        PrefetchPriority::High => "HIGH",
                        PrefetchPriority::Medium => "MED",
                    };
                    PrefetchInstruction {
                        target_layer: e.dst_layer,
                        target_expert: e.dst_expert,
                        priority: e.priority,
                        size_bytes: e.prefetch_size_bytes,
                        comment: format!(
                            "P(L{}:E{} | L{}:E{}) = {:.2} -> {} priority",
                            e.dst_layer,
                            e.dst_expert,
                            e.src_layer,
                            e.src_expert,
                            e.conditional_prob,
                            priority_str,
                        ),
                    }
                })
                .collect();
            // Sort: HIGH priority first
            instructions.sort_by_key(|i| match i.priority {
                PrefetchPriority::High => 0,
                PrefetchPriority::Medium => 1,
            });
            PseudoKernel {
                layer,
                expert,
                prefetch_instructions: instructions,
            }
        })
        .collect();

    kernels.sort_by_key(|k| (k.layer, k.expert));
    kernels
}

/// Render a pseudo-kernel to a human-readable string (Triton-like pseudocode).
pub fn render_pseudo_kernel(kernel: &PseudoKernel) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# Compiler-generated kernel for layer {}, expert {}\n",
        kernel.layer, kernel.expert
    ));
    out.push_str(&format!(
        "@triton.jit\ndef layer{}_expert{}_kernel(input_ptr, output_ptr, expert_weight_ptrs, pq):\n",
        kernel.layer, kernel.expert
    ));
    out.push_str("    # 1. Compute expert output\n");
    out.push_str(&format!(
        "    output = matmul(input_ptr, expert_weight_ptrs[{}][{}])\n\n",
        kernel.layer, kernel.expert
    ));
    out.push_str("    # 2. Statically-emitted prefetches from routing graph\n");

    for instr in &kernel.prefetch_instructions {
        out.push_str(&format!("    # {}\n", instr.comment));
        out.push_str(&format!(
            "    async_prefetch(expert_weight_ptrs[{}][{}], size={}, priority={:?})\n",
            instr.target_layer, instr.target_expert, instr.size_bytes, instr.priority,
        ));
    }

    if kernel.prefetch_instructions.is_empty() {
        out.push_str("    # (no prefetches — last layer or no high-prob edges)\n");
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::passes::prefetch_emitter::{PrefetchEntry, PrefetchPriority, PrefetchSchedule};
    use std::collections::HashMap;

    #[test]
    fn test_generate_pseudo_kernels() {
        let schedule = PrefetchSchedule {
            entries: vec![
                PrefetchEntry {
                    src_layer: 0,
                    src_expert: 0,
                    dst_layer: 1,
                    dst_expert: 0,
                    priority: PrefetchPriority::High,
                    conditional_prob: 0.80,
                    prefetch_size_bytes: 1_000_000,
                    can_batch: false,
                },
                PrefetchEntry {
                    src_layer: 0,
                    src_expert: 0,
                    dst_layer: 1,
                    dst_expert: 1,
                    priority: PrefetchPriority::Medium,
                    conditional_prob: 0.45,
                    prefetch_size_bytes: 500_000,
                    can_batch: false,
                },
            ],
            per_layer_counts: HashMap::from([(0, 2)]),
        };

        let kernels = generate_pseudo_kernels(&schedule);
        assert_eq!(kernels.len(), 1);
        assert_eq!(kernels[0].layer, 0);
        assert_eq!(kernels[0].expert, 0);
        assert_eq!(kernels[0].prefetch_instructions.len(), 2);
        // HIGH priority should be first
        assert_eq!(
            kernels[0].prefetch_instructions[0].priority,
            PrefetchPriority::High
        );
    }

    #[test]
    fn test_render_pseudo_kernel() {
        let kernel = PseudoKernel {
            layer: 5,
            expert: 3,
            prefetch_instructions: vec![PrefetchInstruction {
                target_layer: 6,
                target_expert: 1,
                priority: PrefetchPriority::High,
                size_bytes: 2_000_000,
                comment: "P(L6:E1 | L5:E3) = 0.80 -> HIGH priority".into(),
            }],
        };
        let rendered = render_pseudo_kernel(&kernel);
        assert!(rendered.contains("layer 5, expert 3"));
        assert!(rendered.contains("async_prefetch"));
        assert!(rendered.contains("L6:E1"));
    }
}
