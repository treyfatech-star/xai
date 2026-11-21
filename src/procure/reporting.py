"""Table/figure helpers: every table is saved as CSV and LaTeX, every figure as 300-dpi PNG."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from procure.config import load_settings, path


def _tex_escape(s: object) -> str:
    """Escape LaTeX special characters."""
    out = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("±", r"$\pm$")]:
        out = out.replace(a, b) if a != "\\" else out
    return out.replace("±", r"$\pm$")


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Minimal booktabs table (no jinja2 dependency)."""
    cols = [str(c) for c in df.columns]
    lines = [r"\begin{table}[ht]", r"\centering", f"\\caption{{{_tex_escape(caption)}}}", f"\\label{{{label}}}",
             r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule",
             " & ".join(_tex_escape(c) for c in cols) + r" \\", r"\midrule"]
    for row in df.itertuples(index=False):
        lines.append(" & ".join(_tex_escape(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, name: str, caption: str, out_dir: Path | None = None) -> tuple[Path, Path]:
    """Write <name>.csv and <name>.tex into experiments/results."""
    out_dir = out_dir or path("results_dir")
    csv, tex = out_dir / f"{name}.csv", out_dir / f"{name}.tex"
    df.to_csv(csv, index=False)
    tex.write_text(to_latex(df, caption, f"tab:{name}"))
    return csv, tex


def save_figure(fig: "object", name: str, out_dir: Path | None = None) -> Path:
    """Save a matplotlib figure as PNG at the configured dpi and close it."""
    import matplotlib.pyplot as plt

    out_dir = out_dir or path("results_dir")
    p = out_dir / f"{name}.png"
    fig.savefig(p, dpi=load_settings()["experiments"]["figure_dpi"], bbox_inches="tight")
    plt.close(fig)
    return p
