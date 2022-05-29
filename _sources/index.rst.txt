.. cftool documentation master file, created by
   sphinx-quickstart on Sun May 29 12:36:12 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cftool's documentation!
==================================

cftool (Curve Fitting Tool) is a program to do the curve fitting. It supports curve fitting of implicit equations. 

Project Homepage
----------------

- GitHub: `github.com/haiiliin/cftool <https://github.com/haiiliin/cftool>`_


Downloads
---------

You can download the latest version of Abaqus Executor on `GitHub <https://github.com/haiiliin/cftool/releases/latest>`_.


Main Interface
--------------

This is the main interface of cftool.

.. image:: _static/interface.png
   :width: 100%


Usage
-----

To use this program, you should:

- First, import your data in a **csv** file, by default, the odd columns will be used for x series, the even columns will be used for y axis. For example, if your file has 5 columns: **x1**, **x2**, **x3**, **x4**, **x5**, Then **(x1, y1)** and **(x2, y2)** will be two line series to do the curve fitting, **x5** will be ignored. You can also define it by yourself.

- Choose the type of the equation. You can choose the equation definded in the program, or you can define your own equation. The equation must contain two variable :math:`x` and :math:`y`, other variables will be used for optimization. The equation can be an implicit equation, i.e., :math:`a * x^2 + b * y^2 = 1`. 
- Set the initial coefficients, lower and upper bounds. The default initial coefficient for every coefficient is 1, and the default bounds is :math:`(-\infty, \infty)`.
- Fitting. Click the action **Fit Curve** to do the curve fitting, the results will be showed in the figure. 


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
