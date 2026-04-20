"""R-PGO Codegen: emits executable Triton kernels from compiler artifacts."""

from rpgo.codegen.triton_emitter import TritonKernelEmitter, emit_prefetch_kernels

__all__ = ["TritonKernelEmitter", "emit_prefetch_kernels"]
