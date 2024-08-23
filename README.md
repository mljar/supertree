# SuperTree - Interactive Decision Tree Visualization
Interactive Decision Tree Visualization is a Python package designed to visualize decision trees in an interactive and user-friendly way within Jupyter Notebooks, Jupyter Lab, Google Colab, and any other notebooks that support HTML rendering. The visualizations are powered by JavaScript, primarily using the D3.js library, providing a rich and dynamic experience.
## Description
This package allows users to seamlessly integrate decision tree visualizations into their data analysis workflows. With this tool, you can not only display decision trees, but also interact with them directly within your notebook environment. Key features include the ability to zoom and pan through large trees, collapse and expand specific nodes, and explore the structure of the tree in an intuitive and visually appealing manner.

Whether you're presenting your analysis to others or exploring complex models yourself, this package enhances the way you work with decision trees by making them more accessible and easier to understand.

## Instalation
You can install SuperTree package using pip. To install the package, simply run the following command in your terminal or command prompt.
`pip install supertree`

## Requirements
Before using Interactive Decision Tree Visualization, ensure that the following dependencies are installed. These packages are necessary for the library to function properly:

- pandas: pandas>=2.0.0
- numpy: numpy>=2.0.0
- ipython: ipython>=8.0.0

These dependencies will be installed automatically when you install the package using pip install supertree. However, if you are setting up the environment manually, ensure that these packages are installed with the specified versions or higher.
## Supported Libraries and Models

**Interactive Decision Tree Visualization** currently supports decision tree models from the following popular machine learning libraries:

- **scikit-learn** (`sklearn`)
- **LightGBM**
- **XGBoost**

### Supported Models

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

If we do not support the model you want to use, you can convert it to a supported format, and here is an example of how to do that. For now it is experimental feature we still working on this.
```python
from supertree.model_loader import ModelLoader
from supertree import SuperTree

# This is how the tree_dict list should look. It has been converted from a model that does not support NoneType.
# NoneType values are not allowed, so placeholders are used instead:
# - feature: -1 indicates no feature (used for leaf nodes).
# - threshold: -1 or -2 indicates no threshold (used for leaf nodes).
# - left_child_index and right_child_index: -1 indicates no child (used for leaf nodes).
# class_distribution: must reflect the correct distribution of classes for classification.
# the rest of the data does not have to be correct

tree_dict = [
    {
        "index": 0,
        "feature": 1,
        "impurity": 0.5,
        "threshold": 1.5,
        "class_distribution": [10, 10],
        "predicted_class": 0,
        "samples": 20,
        "is_leaf": False,
        "left_child_index": 1,
        "right_child_index": 2,
    },
    {
        "index": 1,
        "feature": -1,
        "impurity": 0.0,
        "threshold": -1,
        "class_distribution": [10, 0],
        "predicted_class": 0,
        "samples": 10,
        "is_leaf": True,
        "left_child_index": -1,
        "right_child_index": -1,
    },
    {
        "index": 2,
        "feature": -1,
        "impurity": 0.0,
        "threshold": -2,
        "class_distribution": [0, 10],
        "predicted_class": 1,
        "samples": 10,
        "is_leaf": True,
        "left_child_index": -1,
        "right_child_index": -1,
    }
]

my_model = ModelLoader("classification",tree_dict)

st = SuperTree(my_model)
st.show_tree()

```

## Example
```python
Simple Classification Decision Tree Example
from supertree import SuperTree

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()


X, y = iris.data, iris.target

#Train model
model = DecisionTreeClassifier()
model.fit(X, y)

#Create super tree
super_tree = SuperTree(model, X, y, iris.feature_names, iris.target_names)
#You can create SuperTree without feature and target names will be generated automatically
#SuperTree(model, X , y)
#You can also create SuperTree from only model
#super_tree = SuperTree(model)
super_tree.save_html("tree")
#^ Saving html output locally with tree.html name
super_tree.save_json_tree("tree")
#^ Saving json tree locally with tree.json name
super_tree.show_tree()
#^show tree in your notebook
```
Random Forest Regressor Example
```python
from supertree import SuperTree

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

super_tree = SuperTree(model,X, y)
super_tree.show_tree(2)
# In models with forest you can choose witch tree you want to show or save.
```
For more example go to examples directory.
## Support
If you encounter any issues, find a bug, or have a feature request, we would love to hear from you! Please don't hesitate to reach out to us at supertree/issues. We are committed to improving this package and appreciate any feedback or suggestions you may have.
