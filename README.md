![](https://raw.githubusercontent.com/mljar/supertree/main/media/supertree2.gif)

# SuperTree - Interactive Decision Tree Visualization

Interactive Decision Tree Visualization is a Python package designed to visualize decision trees in an **interactive** and user-friendly way within Jupyter Notebooks, Jupyter Lab, Google Colab, and any other notebooks that support HTML rendering. 



## Description

This package allows users to seamlessly integrate decision tree visualizations into their data analysis workflows. With this tool, you can not only display decision trees, but also interact with them directly within your notebook environment. 

## Key features 

Whether you're presenting your analysis to others or exploring complex models yourself, this package enhances the way you work with decision trees by making them more accessible and easier to understand. Key features include:
- ability to zoom and pan through large trees,
- collapse and expand selected nodes, 
- explore the structure of the tree in an intuitive and visually appealing manner.


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

### Supported Algorithms

The package is compatible with a wide range of classifiers and regressors from these libraries, specifically:

#### Scikit-learn
- `DecisionTreeClassifier`
- `ExtraTreeClassifier`
- `ExtraTreesClassifier`
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `DecisionTreeRegressor`
- `ExtraTreeRegressor`
- `ExtraTreesRegressor`
- `RandomForestRegressor`
- `GradientBoostingRegressor`

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

## Examples

### Decision Tree classifier on iris data.

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

### Random Forest Regressor Example
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from supertree import SuperTree  # <- import supertree :)

# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Initialize supertree
super_tree = SuperTree(model,X, y)
# show tree with index 2 in your notebook
super_tree.show_tree(2)
```
There are more code snippets in the [examples](examples) directory.

## Support

If you encounter any issues, find a bug, or have a feature request, we would love to hear from you! Please don't hesitate to reach out to us at supertree/issues. We are committed to improving this package and appreciate any feedback or suggestions you may have.

## License 
