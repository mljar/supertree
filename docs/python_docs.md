# Class: `SuperTree`
## Description
The main class responsible for converting, displaying, and saving HTML models.
### Methods
 - **`__init__(model,feature_data, target_data, feature_names, target_names, license_key)`**
    - **Desription**: The constructor of the SuperTree class validates a large portion of the attributes.
    - **Atributes**:
        - `model` decision tree model required argument
        - `feature_data` Attribute data that can be in the form of a list, np.array, pd.DataFrame, or pd.Series. An optional argument that must be provided along with `target_data`.
        - `taget_data` Attribute data that can be in the form of a list, np.array, pd.DataFrame, or pd.Series. An optional argument that must be provided along with `feature_data`.
        - `feature_names` A list of feature names. An optional argument that must be provided along with the previous arguments and `target_names`.
        - `target_names` A list of target names for classification or single string for regression. An optional argument that must be provided along with the previous arguments and `feature_names`.
        - `license_key` An optional argument unlocking premium features.
 - **`show_tree(which_tree,which_iteration, start_depth, max_samples, show_sample)`** 
    - **Description:** A method that displays an HTML file in a notebook using IPython, which   contains the SuperTree model.
    - **Atributes:**
        - `which_tree` *(int)*  An optional attribute allowing the selection of a specific tree from a Random Forest or gradient-based models. Defaults to 0.
        - `which_iteration` *(int)* An optional attribute working only in HistGradientBoosting model allowing selection of specific iteration. Default to 0.
        - `start_depth` *(int)* An optional attribute that sets the initial display height of the tree. Useful for large trees that take time to render. Deafault to 5.
        - `max_samples` *(int)* An optional attribute that sets the maximum number of data points to be used for visualization. Useful when dealing with large datasets, especially in regression, where displaying tens of thousands of samples can take several seconds. Default is set to 7,500.
         At the moment, in scikit-learn trees, the  smaller data can be inconsistent, especially in the leaves, because the model stores the base distribution of the data within the model.
        - `show_sample` *(list(int))* An optional argument that only works with a valid license. It adds a button to the visualization that shows the path of a given sample in the tree.
 - **`save_html(filename,which_tree, which_iteration, start_depth, max_samples, show_sample)`**
    - **Description:** A method that save html file with SuperTree model.
    - **Atributes:**
        - `filename` *(string)* The file name under which the HTML will be saved. Defaultss set to output.html.
        - `which_tree` *(int)*  An optional attribute allowing the selection of a specific tree from a Random Forest or gradient-based models. Defaults to 0.
        - `which_iteration` *(int)* An optional attribute working only in HistGradientBoosting model allowing selection of specific iteration. Default to 0.
        - `start_depth` *(int)* An optional attribute that sets the initial display height of the tree. Useful for large trees that take time to render. Deafault to 5.
        - `max_samples` *(int)* An optional attribute that sets the maximum number of data points to be used for visualization. Useful when dealing with large datasets, especially in regression, where displaying tens of thousands of samples can take several seconds. Default is set to 7,500.
         At the moment, in scikit-learn trees, the  smaller data can be inconsistent, especially in the leaves, because the model stores the base distribution of the data within the model.
        - `show_sample` *(list(int))* An optional argument that only works with a valid license. It adds a button to the visualization that shows the path of a given sample in the tree.
 - **`save_json_tree(filename, which_tree, which_iteration, max_samples, show_sample)`**
     - **Description** A method that save tree data in json format.
     - **Atributes:**
        - `filename` *(string)* The file name under which the Json will be saved. Default set to treedata.json.
        - `which_tree` *(int)*  An optional attribute allowing the selection of a specific tree from a Random Forest or gradient-based models. Defaults to 0.
        - `which_iteration` *(int)* An optional attribute working only in HistGradientBoosting model allowing selection of specific iteration. Default to 0.
        - `start_depth` *(int)* An optional attribute that sets the initial display height of the tree. Useful for large trees that take time to render. Deafault to 5.
        - `max_samples` *(int)* An optional attribute that sets the maximum number of data points to be used for visualization. Useful when dealing with large datasets, especially in regression, where displaying tens of thousands of samples can take several seconds. Default is set to 7,500.
         At the moment, in scikit-learn trees, the  smaller data can be inconsistent, especially in the leaves, because the model stores the base distribution of the data within the model.
        - `show_sample` *(list(int))* An optional argument that only works with a valid license. It adds a button to the visualization that shows the path of a given sample in the tree.
 - **`_get_combined_data()`**: 
    - **Description**: A private method that combines and return two variables in JSON format into one: data describing the entire tree and data concerning individual nodes.
 - **`_get_json_tree_data()`**:
    - **Description:**: A private method that convert and return TreeData object into JSON format.
 - **`_get_json_nodes()`** 
     - **Description:** A private method that convert Nodes objects into JSON format.
 - **`_create_node_dfs(node_index, left_right, threshold, feature, x_axis)`**: 
     - **Description:** A private method that uses the DFS algorithm to create a tree from an array of nodes. (The parent node points to the child nodes)."
     - **Atributes:**
        - `node_index`: (int) Index of node
        - `left_right`: (string) Describe it is a left or right node of the parent.
        - `threshold`: (float) Threshold for parent node.
        - `feature`: (int) Feature of parent node.
        - `x_axis`: (list(float)) Limiting samples in the parent node
 - **`_which_model()`**:
      - **Description:** A private method that identifies the model ("classification" or "regression)" based on the class name or the methods it possesses.
 - **`_count_class_distribution(node)`**: 
      - **Description:** A private method that counts the number of samples for each class in the entire tree for a given node using the DFS algorithm in each node, if the model does not contain this information.
       - **Atributes:**
            - `node`: Current node in dfs. 
 - **`_is_model_fitted()`**:
      - **Description:** A private method that checks if a given model has been trained, and if not, raises an exception.
 - **`_count_samples()`**: 
    - **Description:**  A private method that counts the total number of samples passing to a given node using the DFS algorithm.
 - **`_convert_model_to_dict_array()`**
    - **Description:** A private method that converts the data contained in the model into the format shown below.
 ```
                 node_info = {
                    "index": ,
                    "feature": ,
                    "impurity":,
                    "threshold":,
                    "class_distribution": ,
                    "predicted_class": ,
                    "samples": ,
                    "is_leaf": ,
                    "left_child_index":,
                    "right_child_index":,
                }
 ```
  - **`_collect_node_info_lgbm(node,depth)`** A private method that converts the data contained in the lgbm models.
    - **Atributes:**
        - `node`: Current node in dfs.
        - `depth`: Current depth in dfs.
   - **`_collect_node_info_xgboost(node,depth)`** A private method that converts the data contained in the xgb models.
     - **Atributes:**
        - `node`: Current node in dfs.
        - `depth`: Current depth in dfs.
    - **`_collect_node_info_hist(nodes)`** 
        - **Description** A private method that converts the data contained in the HistGradientBoosting models.
        - **Atributes:**
            - `nodes`: List of node data.
    - **`_collect_node_info_onnx(onnx_model)`**
        - **Description** A private method that converts the data contained in the onnx models. 
        - **Atributes** 
            - `onnx_model`: Tree model from onnx 

# Class: `Node`
## Description
The class represents one node
### Methods
- **` __init__(feature, threshold, impurity, samples,                 class_distribution, treeclass, is_leaf,left_children, right_children):`**
    - **Attributes**:
        - `feature` *(int)*  Index of feature.
        - `threshold` *(float)* Threshold.
        - `impurity` *(float)* Gini, entropy etc
        - `class_distribution` *(float)* * List of integers
        - `tree_class` *(int)* Predicted class in specific node.
        - `is_leaf` *(bool)*
        - `left_children` *(int)* Index of left child
        - `right_children`*(int)* Index of right child
- **`to_dict()`**
    - **Decription** Convert node values to dictonary that is JSON serializable.
- **`add_left(left_node)`**
    - **Description:** Adding a reference to the right child of the Node object.
- **`add_right(right_node`)**
    - **Description:** Adding a reference to the left child of the Node object.
# Class: `TreeData`
## Description
The class represents data for tree or forest.
### Methods
- **`__init__(
        self, tree_type, feature_names, target_names, data_feature, data_target
    )`**:
    - **Atributes** Same as SuperTree Class.
- **`to_dict()`**
     - **Decription** Convert tree data to dictonary that is JSON serializable.
- **`extract_values_if_dataframe(data)`**
    - **Description:** Covert dataframe to list.
- **convert_target_strings(data_target)**
    - **Description:** If target data is names not index coverting into indexes.
- **`set_show_sample(sample)`**
    - **Description:** Setting show_sample variable.
- **`reset_sample()`**
    - **Description:** Reset show_sample variable.
 # Class: `TreeData`
## Description
Class for converting not supported trees.
### Methods
- **`__init__(model_type, model_dict_list)`**
    - **Attributes:**
        - `model_type` *(string)* Classification or Regression
        - `model_dict_list` Nodes in dictonary list in format shown below. Every value must be not None but not every value has to be correct to see a proper visualization. .
```
                 node_info = {
                    "index": (int) index of node (Optional) ,
                    "feature": (int) index of feature (Not optional),
                    "impurity":(float) Gini, entropy etc... (Optional),
                    "threshold":(float) value of threshold (Not optional),
                    "class_distribution": (list(int)) lA list of the number of samples from each class, the true value is not required in regression, a list with a single value is sufficient, e.g., [[10]],
                    "predicted_class": (string) name of predicted class not optional in leafs  ,
                    "samples": (int) Value of samples in node,
                    "is_leaf": (bool) Bool value of is a leaf or not,
                    "left_child_index": (int) index of left child,
                    "right_child_index":(int) index of right child,
                }
 ```

`





