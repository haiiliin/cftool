import typing

import numpy as np
import pandas as pd
from PyQt5.QtCore import QStandardPaths, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QTableWidgetItem, QAction, QAbstractItemView, QCompleter
from matplotlib import rcParams
from scipy import optimize
from scipy.optimize import Bounds
from sympy import Basic, Expr, Add, Mul, lambdify, S, Symbol, Eq

from .ui_mainwindow import Ui_MainWindow

config = {
    "font.family": 'serif',
    # "font.size": 10,
    "mathtext.fontset": 'stix',
}
rcParams.update(config)


class MainWindow(QMainWindow):
    data: pd.DataFrame
    coefs: list[Basic]
    coef_values: list[list[float]]
    coef_rmses: list[float]
    coef_initials: list[float]
    coef_bounds: Bounds
    f: Expr
    func: typing.Optional[typing.Callable]
    xcols: list[str]
    ycols: list[str]
    customEquation: str

    importAction: QAction
    fitAction: QAction

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon('icons/curve-fitting.png'))
        self.setMinimumSize(1000, 800)
        self.setupActions()

        # Attributes
        self.data = pd.DataFrame()
        self.coefs = []
        self.coef_values = []
        self.coef_rmses = []
        self.coef_initials = []
        self.coef_bounds = Bounds(-np.inf, np.inf)
        self.f = Expr()
        self.func = None
        self.xcols = []
        self.ycols = []
        self.customEquation = self.ui.equation.text()

        self.parseEquation()

    def setupActions(self):
        self.importAction = QAction()
        self.importAction.setText('Import')
        self.importAction.setIcon(QIcon('icons/import-csv.png'))
        self.importAction.triggered.connect(self.importData)

        self.fitAction = QAction()
        self.fitAction.setText('Fit Curve')
        self.fitAction.setIcon(QIcon('icons/process.png'))
        self.fitAction.triggered.connect(self.fit)

        self.ui.toolBar.addActions([self.importAction, self.fitAction])

        self.ui.type.currentTextChanged.connect(self.setupRegularEquation)
        self.ui.equation.textChanged.connect(self.updateCustomEquation)
        self.ui.equation.textChanged.connect(self.parseEquation)

    def setupRegularEquation(self, text: str):
        if text == 'Custom':
            self.ui.equation.setText(self.customEquation)
        elif text == 'Polynomial':
            self.ui.equation.setText('y = a * x^2 + b * x + c')
        elif text == 'Exponential':
            self.ui.equation.setText('y = a * exp(b * x) + c')
        elif text == 'Ellipse':
            self.ui.equation.setText('x^2 / a^2 + y^2 / b^2 = 1')

    def updateCustomEquation(self, text: str):
        if self.ui.type.currentText() == 'Custom':
            self.customEquation = text

    def test(self):
        self.data = pd.read_csv('tests/data.csv')
        self.ui.dataTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ui.dataTable.setRowCount(self.data.shape[0])
        self.ui.dataTable.setColumnCount(self.data.shape[1])
        self.ui.dataTable.setHorizontalHeaderLabels(self.data.columns)
        for row in range(self.data.shape[0]):
            for col in range(self.data.shape[1]):
                item = QTableWidgetItem(f'{self.data.iloc[row, col]:.4f}')
                self.ui.dataTable.setItem(row, col, item)
        self.xcols = ['x1', 'x2', 'x3', 'x4', 'x5']
        self.ycols = ['y1', 'y2', 'y3', 'y4', 'y5']
        self.fit()

    def importData(self):
        documentationPath = QStandardPaths.standardLocations(QStandardPaths.DocumentsLocation)[0]
        filePath, _ = QFileDialog.getOpenFileName(self, 'Select File', documentationPath,
                                                  'Comma-Separated File (*.csv)')
        self.data = pd.read_csv(filePath)
        self.ui.dataTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ui.dataTable.setRowCount(self.data.shape[0])
        self.ui.dataTable.setColumnCount(self.data.shape[1])
        self.ui.dataTable.setHorizontalHeaderLabels(self.data.columns)
        for row in range(self.data.shape[0]):
            for col in range(self.data.shape[1]):
                item = QTableWidgetItem(f'{self.data.iloc[row, col]:.4f}')
                self.ui.dataTable.setItem(row, col, item)

        xcols = list(self.data.columns[0::2])
        ycols = list(self.data.columns[1::2])
        if len(xcols) > len(ycols):
            xcols = xcols[:-1]
        self.xcols, self.ycols = xcols, ycols
        self.ui.xs.setText(', '.join(self.xcols))
        self.ui.ys.setText(', '.join(self.ycols))

        completer = QCompleter(self.data.columns)
        completer.setFilterMode(Qt.MatchEndsWith)
        self.ui.xs.setCompleter(completer)
        self.ui.ys.setCompleter(completer)

    def parseEquation(self):
        equation = self.ui.equation.text()
        try:
            expr = Eq(*map(S, equation.split('=')))
            self.coefs = list(expr.free_symbols)
            self.coefs.remove(Symbol('x'))
            self.coefs.remove(Symbol('y'))
            self.coef_initials = [1] * len(self.coefs)
            self.f = Add(Mul(expr.args[0], -1), expr.args[1])
            self.func = lambdify([Symbol('x'), Symbol('y')] + self.coefs, self.f)
        except Exception:
            return

        coef_list = ', '.join([str(coef) for coef in self.coefs])
        self.ui.initial.setPlaceholderText(f'Initial values for {coef_list}')
        self.ui.initial.setToolTip(f'Initial values for {coef_list}')
        self.ui.lower.setPlaceholderText(f'Lower bounds for {coef_list}')
        self.ui.lower.setToolTip(f'Lower bounds for {coef_list}')
        self.ui.upper.setPlaceholderText(f'Upper bounds for {coef_list}')
        self.ui.upper.setToolTip(f'Upper bounds for {coef_list}')

        def process_values(text: str, default_value: float = 1.0) -> list[float]:
            items = text.replace(' ', '').split(',')
            if len(items) == 1:
                try:
                    values = [float(items[0])] * len(self.coefs)
                except ValueError:
                    values = [default_value] * len(self.coefs)
            else:
                values = []
                for i in range(len(self.coefs)):
                    if i >= len(items):
                        values.append(default_value)
                    else:
                        try:
                            values.append(float(items[i]))
                        except ValueError:
                            values.append(default_value)
            return values

        self.coef_initials = process_values(self.ui.initial.text())
        self.ui.initial.setText(', '.join([str(item) for item in self.coef_initials]))
        lb = process_values(self.ui.lower.text(), -np.inf)
        ub = process_values(self.ui.upper.text(), np.inf)
        self.ui.lower.setText(', '.join([str(item) for item in lb]))
        self.ui.upper.setText(', '.join([str(item) for item in ub]))
        self.coef_bounds = Bounds(lb, ub)

        xcols = self.ui.xs.text().replace(' ', '').split(',')
        ycols = self.ui.ys.text().replace(' ', '').split(',')
        self.xcols, self.ycols = [], []
        for xcol in xcols:
            if xcol in self.data.columns:
                self.xcols.append(xcol)
        for ycol in ycols:
            if ycol in self.data.columns:
                self.ycols.append(ycol)
        if len(self.xcols) != len(self.ycols):
            raise ValueError('Length of xs and ys are not equal')
        self.ui.xs.setText(', '.join(self.xcols))
        self.ui.ys.setText(', '.join(self.ycols))

    def rmse(self, coefs: typing.Iterable[float], x, y):
        return np.linalg.norm(self.func(x, y, *tuple(coefs))) / np.sqrt(len(x))

    def fit(self):
        self.parseEquation()
        self.coef_values = []
        self.coef_rmses = []
        for xcol, ycol in zip(self.xcols, self.ycols):
            res = optimize.minimize(self.rmse, np.array(self.coef_initials), args=(self.data[xcol], self.data[ycol]),
                                    bounds=self.coef_bounds, method='L-BFGS-B')
            self.coef_values.append(list(res.x))
            self.coef_rmses.append(res.fun)
        self.plot()

    @property
    def mpl(self):
        return self.ui.mpl

    @property
    def ax(self):
        return self.mpl.ax

    @property
    def fig(self):
        return self.mpl.fig

    def updateFigure(self):
        self.ax.axis('equal')
        self.fig.tight_layout()
        self.ax.grid()
        self.ui.mpl.draw()

    def plot(self):
        self.mpl.clear()

        xlb, xub = np.inf, -np.inf
        ylb, yub = np.inf, -np.inf
        colors = [f'C{i + 1}' for i in range(len(self.xcols))]
        for xcol, ycol, color, coef_value, rmse, i in zip(self.xcols, self.ycols, colors,
                                                          self.coef_values, self.coef_rmses, range(len(self.xcols))):
            x, y = self.data[xcol], self.data[ycol]
            label = f'{xcol}-{ycol}, '
            label += ', '.join(['{} = {:.4f}'.format(coef, value) for coef, value in zip(self.coefs, coef_value)])
            label += f', RMSE = {rmse:.4f}'
            self.ax.scatter(x, y, color=color, label=label)
            if np.min(x) < xlb:
                xlb = np.min(x)
            if np.max(x) > xub:
                xub = np.max(x)
            if np.min(y) < ylb:
                ylb = np.min(y)
            if np.max(y) > yub:
                yub = np.max(y)

        xlen, ylen = xub - xlb, yub - ylb
        xrange = np.linspace(xlb - xlen * 0.5, xub + xlen * 0.5, 100)
        yrange = np.linspace(ylb - ylen * 0.5, yub + ylen * 0.5, 100)
        X, Y = np.meshgrid(xrange, yrange)

        for coef_value, color in zip(self.coef_values, colors):
            Z = self.func(X, Y, *tuple(coef_value))
            self.ax.contour(X, Y, Z, [0], colors=color)

        self.ax.legend()
        self.updateFigure()
