![](https://raw.githubusercontent.com/mljar/supertree/main/media/supertree2.gif)


# `supertree` - Interactive Decision Tree Visualization

[![PyPI version](https://badge.fury.io/py/supertree.svg)](https://pypi.org/project/supertree/)
[![Downloads](https://img.shields.io/pypi/dm/supertree)](https://pypi.org/project/supertree/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.txt)


Visualize decision trees interactively in Jupyter, JupyterLab, and Google Colab.
Zoom, pan, collapse nodes, and trace sample paths - all inside your notebook.

Works with scikit-learn, XGBoost, LightGBM, and ONNX.

## Installation

```bash
pip install supertree
```

## Quick Start

### Visualize Decision Tree classifier on iris data 

<a target="_blank" href="https://colab.research.google.com/drive/1f2Xu8CwbXaT33hvh-ze0JK3sBSpXBt5T?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from supertree import SuperTree

# Load the iris dataset
iris = load_iris()

# Train model
model = DecisionTreeClassifier(max_depth=3)
model.fit(iris.data, iris.target)

# Initialize supertree
super_tree = SuperTree(model, iris.data, iris.target, iris.feature_names, iris.target_names)

# show tree in your notebook
super_tree.show_tree()
```

![](https://raw.githubusercontent.com/mljar/supertree/main/media/supertree-decision-tree-visualization.png)

### It works with trees ensembles too - Random Forest Regressor Example

<a target="_blank" href="https://colab.research.google.com/drive/1nR7GlrIKcMQYdnMm_duY7a6vscyqTCMj?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from supertree import SuperTree  # <- import supertree :)

# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
model.fit(X, y)

# Initialize supertree
super_tree = SuperTree(model,X, y)
# show tree with index 2 in your notebook
super_tree.show_tree(2)
```

![](https://raw.githubusercontent.com/mljar/supertree/main/media/supertree-random-forest-visualization.png)

There are more code snippets in the [examples](examples) directory.


## Supported Libraries

- **scikit-learn** (`sklearn`)
- **LightGBM**
- **XGBoost**
- **ONNX**:

### Supported Algorithms

The package is compatible with a wide range of classifiers and regressors from these libraries, specifically:

#### Scikit-learn
- `DecisionTreeClassifier`
- `ExtraTreeClassifier`
- `ExtraTreesClassifier`
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `HistGradientBoostingClassifier`
- `DecisionTreeRegressor`
- `ExtraTreeRegressor`
- `ExtraTreesRegressor`
- `RandomForestRegressor`
- `GradientBoostingRegressor`
- `HistGradientBoostingRegressor`

#### LightGBM
- `LGBMClassifier`
- `LGBMRegressor`
- `Booster`

#### XGBoost
- `XGBClassifier`
- `XGBRFClassifier`
- `XGBRegressor`
- `XGBRFRegressor`
- `Booster`

If we do not support the model you want to use, please let us know.


## Articles

- [Visualize decision tree from scikit-learn package](https://mljar.com/blog/visualize-decision-tree/)
- [4 ways to vizualize decision tree from LightGBM](https://mljar.com/blog/visualize-lightgbm-tree/)
- [How to visualize decision tree from Xgboost](https://mljar.com/blog/visualize-xgboost-tree/)


## Support

If you encounter any issues, find a bug, or have a feature request, we would love to hear from you! Please don't hesitate to reach out to us at supertree/issues. We are committed to improving this package and appreciate any feedback or suggestions you may have.

## License 

`supertree` is open source under the Apache License 2.0.
