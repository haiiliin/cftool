import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class MatplotlibWidget(QWidget):
    fig: Figure
    ax: Axes
    canvas: FigureCanvas

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Layout
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

    def draw(self):
        self.canvas.draw()
        self.fig.tight_layout()

    def clear(self):
        self.ax.clear()
