"""PDF renderer for tearsheet reports.

Converts HTML tearsheet output to PDF using weasyprint when available.
Falls back to a simple matplotlib-based multi-page PDF otherwise.
"""

from __future__ import annotations

import importlib

__all__ = ["render_pdf"]


def _weasyprint_available() -> bool:
    """Check whether weasyprint is installed.

    Returns:
        True if weasyprint can be imported.
    """
    try:
        importlib.import_module("weasyprint")
        return True
    except ImportError:
        return False


def render_pdf(
    html_content: str,
    output_path: str,
) -> None:
    """Render an HTML tearsheet to PDF.

    Uses weasyprint if available; otherwise falls back to a minimal
    matplotlib-based PDF with a notice that full rendering requires
    weasyprint.

    Args:
        html_content: Complete HTML document string.
        output_path: File path for the output PDF.

    Raises:
        RuntimeError: If neither weasyprint nor matplotlib can produce output.
    """
    if _weasyprint_available():
        import weasyprint  # type: ignore[import-untyped]

        doc = weasyprint.HTML(string=html_content)
        doc.write_pdf(output_path)
        return

    _fallback_pdf(output_path)


def _fallback_pdf(output_path: str) -> None:
    """Create a minimal fallback PDF using matplotlib.

    Args:
        output_path: File path for the output PDF.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(
            0.5, 0.6,
            "QuantLite Tearsheet",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=20, fontweight="bold",
        )
        ax.text(
            0.5, 0.45,
            "Install weasyprint for full PDF rendering:\n"
            "pip install quantlite[pdf]",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=12, color="#666",
        )
        pdf.savefig(fig)
        plt.close(fig)
