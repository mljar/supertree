![](https://raw.githubusercontent.com/mljar/supertree/main/media/supertree2.gif)

# `supertree` - Interactive Decision Tree Visualization

`supertree` is a Python package designed to visualize decision trees in an **interactive** and user-friendly way within Jupyter Notebooks, Jupyter Lab, Google Colab, and any other notebooks that support HTML rendering. With this tool, you can not only display decision trees, but also interact with them directly within your notebook environment. Key features include:
- ability to zoom and pan through large trees,
- collapse and expand selected nodes, 
- explore the structure of the tree in an intuitive and visually appealing manner.

## Features

<div style="overflow: hidden;">
  <table style="table-layout: fixed; width: 100%; position: absolute;'">
  <tr>
    <td><img src="https://github.com/mljar/supertree/blob/main/media/videos/2_regression_details-ezgif.com-video-to-gif-converter.gif" alt="Gif1" width="375"/><br/>See all the details</td>
    <td><img src="https://github.com/mljar/supertree/blob/main/media/videos/1_supertree_zoom_an_reset-ezgif.com-video-to-gif-converter.gif" alt="Gif2" width="375"/><br/>Zoom</td>
  </tr>
  </table>
  <table style="table-layout: fixed; width: 100%; position: absolute;'">
    <tr>
    <td><img src="https://github.com/mljar/supertree/blob/main/media/videos/4_fullscreen-ezgif.com-video-to-gif-converter.gif" alt="Gif3" width="375"/><br/>Fullscreen in Jupyter</td>
    <td><img src="https://github.com/mljar/supertree/blob/main/media/videos/6_change_depth_dynamicaly-ezgif.com-video-to-gif-converter.gif" alt="Gif4" width="375"/><br/>Depth change</td>
  </tr>
    </table>
   <table style="table-layout: fixed; width: 100%; position: absolute;'">
    <tr>
    <td><img src="https://github.com/mljar/supertree/blob/main/media/videos/change_palette.gif" alt="Gif5" width="375"/><br/>Color change</td>
    <td><img src="https://github.com/mljar/supertree/blob/main/media/videos/switch_tree_in_forest.gif" alt="Gif6" width="375"/><br/>Navigate in forest</td>
  </tr>
    </table>
   <table style="table-layout: fixed; width: 100%; position: absolute;'">
    <tr>
    <td><img src="https://github.com/mljar/supertree/blob/main/media/videos/sample_path.gif" alt="Gif7" width="375"/><br/>Show specific sample path</td>
    <td><img src="https://github.com/mljar/supertree/blob/main/media/videos/save_svg.gif" alt="Gif8" width="375"/><br/>Save tree to svg</td>
  </tr>
    </table>
  <table style="table-layout: fixed; width: 100%; position: absolute;'">
    <tr>
    <td><img src="https://github.com/mljar/supertree/blob/main/media/videos/3_amount_of_sample_visualized-ezgif.com-video-to-gif-converter.gif" alt="Gif11" width="375"/><br/>Links sample visualization</td>
    <td><img src="https://github.com/mljar/supertree/blob/main/media/videos/7_path_to_leaf-ezgif.com-video-to-gif-converter.gif" alt="Gif12" width="375"/><br/>Showing the path to the leaf</td>
  </tr>
    </table>
</div>

Check this features in example directory :)

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

## Articles

- [Visualize decision tree from scikit-learn package](https://mljar.com/blog/visualize-decision-tree/)
- [4 ways to vizualize decision tree from LightGBM](https://mljar.com/blog/visualize-lightgbm-tree/)
- [How to visualize decision tree from Xgboost](https://mljar.com/blog/visualize-xgboost-tree/)


## Support

If you encounter any issues, find a bug, or have a feature request, we would love to hear from you! Please don't hesitate to reach out to us at supertree/issues. We are committed to improving this package and appreciate any feedback or suggestions you may have.

## License 

`supertree` is a commercial software with two licenses available:

- AGPL-3.0 license
- Commercial license with support and maintenance included. Pricing website https://mljar.com/supertree/ License [supertree-commercial-license.pdf](supertree-commercial-license.pdf).

