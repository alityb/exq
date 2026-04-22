"""Dense Python-side compiler helpers."""

from exq.compiler.dense_quant_planner import DenseQuantPlan, HeadQuantPlan, compute_thresholds, plan_dense_quant

# joint_scheduler requires ortools (optional ILP dependency).
# Guard the import so the rest of the package works on a clean install.
try:
    from exq.compiler.joint_scheduler import JointScheduleResult, solve_joint_schedule
    _ILP_AVAILABLE = True
except ImportError:
    _ILP_AVAILABLE = False
    JointScheduleResult = None  # type: ignore[assignment,misc]
    solve_joint_schedule = None  # type: ignore[assignment]

__all__ = [
    "DenseQuantPlan",
    "HeadQuantPlan",
    "JointScheduleResult",
    "compute_thresholds",
    "plan_dense_quant",
    "solve_joint_schedule",
]
