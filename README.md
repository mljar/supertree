![](https://raw.githubusercontent.com/mljar/supertree/main/media/supertree2.gif)

# `supertree` - Interactive Decision Tree Visualization

`supertree` is a Python package designed to visualize decision trees in an **interactive** and user-friendly way within Jupyter Notebooks, Jupyter Lab, Google Colab, and any other notebooks that support HTML rendering. With this tool, you can not only display decision trees, but also interact with them directly within your notebook environment. Key features include:
- ability to zoom and pan through large trees,
- collapse and expand selected nodes, 
- explore the structure of the tree in an intuitive and visually appealing manner.

## Examples

### Decision Tree classifier on iris data 

<a target="_blank" href="https://colab.research.google.com/drive/1f2Xu8CwbXaT33hvh-ze0JK3sBSpXBt5T?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from supertree import SuperTree # <- import supertree :)

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Initialize supertree
super_tree = SuperTree(model, X, y, iris.feature_names, iris.target_names)

# show tree in your notebook
super_tree.show_tree()
```

![](https://raw.githubusercontent.com/mljar/supertree/main/media/classifier.png)

### Random Forest Regressor Example

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

![](https://raw.githubusercontent.com/mljar/supertree/main/media/regressor.png)

There are more code snippets in the [examples](examples) directory.



## Instalation
You can install SuperTree package using pip:

```
pip install supertree
```

Conda support coming soon.

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

## Support

If you encounter any issues, find a bug, or have a feature request, we would love to hear from you! Please don't hesitate to reach out to us at supertree/issues. We are committed to improving this package and appreciate any feedback or suggestions you may have.

## License 

`supertree` is a commercial software with two licenses available:

- Free for non-commercial purposes such as teaching, academic research, and evaluation. [supertree-non-commercial-license.pdf](supertree-non-commercial-license.pdf).
- Commercial license with support and maintenance included. [supertree-commercial-license.pdf](supertree-commercial-license.pdf).

