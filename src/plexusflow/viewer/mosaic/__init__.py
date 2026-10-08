"""plexusflow.viewer.mosaic — small Qt helpers used by the viewer.

Currently holds ``show_figure``, a self-contained matplotlib→Qt bridge
that pops a figure in a QDialog without touching matplotlib's backend
manager (which is brittle inside a running napari session).
"""
from __future__ import annotations

import warnings as _warnings

_warnings.filterwarnings('ignore', '.*FigureCanvasAgg is non-interactive.*')
_warnings.filterwarnings('ignore', '.*Matplotlib is currently using agg.*')


def show_figure(fig=None, position=None, margin: int = 12):
    """Show a matplotlib Figure in a Qt QDialog.

    Works reliably inside napari regardless of matplotlib backend state
    by wrapping the figure in a ``FigureCanvasQTAgg`` and adding it to a
    QDialog directly — no ``pyplot.show`` / manager involvement.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Defaults to ``plt.gcf()``.
    position : {'top-right', None}, optional
        If ``'top-right'``, the dialog is moved to the top-right corner
        of the primary screen after being shown.
    margin : int
        Pixel gap from the screen edge when ``position`` is set.
    """
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.gcf()

    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from qtpy.QtWidgets import (QDialog, QVBoxLayout, QApplication,
                                       QScrollArea)
        from qtpy.QtCore import Qt

        canvas = FigureCanvasQTAgg(fig)
        # Lock canvas to the figure's natural pixel size so QScrollArea
        # actually scrolls (instead of squeezing the figure to fit).
        fig_w = int(fig.get_figwidth() * fig.dpi)
        fig_h = int(fig.get_figheight() * fig.dpi)
        canvas.setMinimumSize(fig_w, fig_h)
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setWidget(canvas)
        dialog = QDialog()
        dialog.setWindowTitle(fig._suptitle.get_text()
                              if fig._suptitle else 'Figure')
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMinMaxButtonsHint)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        # Cap initial dialog size to fit on-screen; QScrollArea handles
        # the overflow.  Small margin so the window chrome doesn't get
        # cut off by the dock/menu bar.
        try:
            screen = QApplication.primaryScreen()
            avail = screen.availableGeometry() if screen is not None else None
        except Exception:
            avail = None
        if avail is not None:
            max_w = max(400, avail.width() - 60)
            max_h = max(300, avail.height() - 120)
        else:
            max_w, max_h = fig_w, fig_h
        dialog.resize(min(fig_w + 24, max_w), min(fig_h + 24, max_h))
        canvas.draw()
        dialog.show()

        if position == 'top-right':
            try:
                screen = QApplication.primaryScreen()
                if screen is not None:
                    avail = screen.availableGeometry()
                    try:
                        QApplication.processEvents()
                    except Exception:
                        pass
                    fw = dialog.frameGeometry().width()
                    x = avail.x() + avail.width() - fw - margin
                    y = avail.y() + margin
                    dialog.move(int(x), int(y))
            except Exception:
                pass

        if not hasattr(show_figure, '_dialogs'):
            show_figure._dialogs = []
        show_figure._dialogs = [d for d in show_figure._dialogs
                                if d.isVisible()]
        show_figure._dialogs.append(dialog)

    except ImportError:
        try:
            fig.show()
        except Exception:
            plt.show(block=False)


__all__ = ["show_figure"]
