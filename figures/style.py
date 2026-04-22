"""Shared matplotlib style for all ExQ paper figures."""
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Palette ───────────────────────────────────────────────────────────────────
TEAL  = "#5DCAA5"   # ExQ / improvement series
CORAL = "#E07B6A"   # baseline series
BLUE  = "#7EB8D4"   # third series (AWQ etc.)
AMBER = "#E8B84B"   # fourth series
GRAY  = "#B4B2A9"   # neutral
DARK  = "#333333"   # text / annotation


def apply_style() -> None:
    """Apply consistent rcParams to all ExQ figures."""
    mpl.rcParams.update({
        # Fonts
        "font.family":        "sans-serif",
        "font.size":          11,
        # Spines
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        # Grid
        "axes.grid":          True,
        "axes.grid.axis":     "y",
        "grid.color":         "#e0e0e0",
        "grid.linestyle":     "--",
        "grid.linewidth":     0.8,
        "axes.axisbelow":     True,
        # Backgrounds
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        # Legend
        "legend.frameon":     False,
        "legend.fontsize":    10,
        # Ticks
        "xtick.bottom":       True,
        "ytick.left":         True,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        # Output
        "figure.dpi":         150,   # screen preview
        "savefig.dpi":        300,   # publication
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.1,
    })


def bar_label(ax, bar, text: str, offset: float = 0.04,
              fontsize: float = 9.5, color: str = DARK) -> None:
    """Place a text label above a single bar."""
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + offset,
        text,
        ha="center", va="bottom",
        fontsize=fontsize, color=color,
    )


def save(fig, path: str) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
