import json
import re
import sys
from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from IPython.display import HTML, display

import supertree.templatehtml as templatehtml
from supertree.node import Node
from supertree.treedata import TreeData

import importlib.metadata


class SuperTree:
    def __init__(
        self,
        model: object,
        feature_data: Optional[Union[List, np.ndarray,
                                     pd.DataFrame, pd.Series]] = None,
        target_data: Optional[Union[List, np.ndarray,
                                    pd.DataFrame, pd.Series]] = None,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[Union[str, List[str]]] = None,
        licence_key: str = "key",
    ):

        valid_model_classes = [
            "DecisionTreeClassifier",
            "ExtraTreeClassifier",
            "ExtraTreesClassifier",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "DecisionTreeRegressor",
            "ExtraTreeRegressor",
            "ExtraTreesRegressor",
            "RandomForestRegressor",
            "GradientBoostingRegressor",
            "Booster",
            "LGBMClassifier",
            "LGBMRegressor",
            "XGBoostBooster", 
            "XGBClassifier", 
            "XGBRegressor", 
            "XGBRFClassifier", 
            "XGBRFRegressor",
            "ModelLoader"
        ]

        if model.__class__.__name__ not in valid_model_classes:
            raise TypeError(
                f"Invalid model type. Expected one of {valid_model_classes}, but got {type(model).__name__}"
            )

        if feature_data is not None:
            if not isinstance(feature_data, (list, np.ndarray, pd.DataFrame, pd.Series)):
                raise TypeError(
                    "Invalid feature_data type. Expected a list, numpy array, or pandas DataFrame."
                )

        if target_data is not None:
            if not isinstance(target_data, (list, np.ndarray, pd.DataFrame, pd.Series)):
                raise TypeError(
                    "Invalid target_data type. Expected a list, numpy array, or pandas DataFrame."
                )

        if target_names is not None:
            if isinstance(target_names, (str, list)):
                if isinstance(target_names, list) and not all(isinstance(name, str) for name in target_names):
                    raise TypeError(
                        "Invalid target_names type. Expected a list of strings.")
            elif not isinstance(target_names, (np.ndarray)):
                raise TypeError(
                    "Invalid target_names type. Expected a string, list of strings, numpy array, or pandas DataFrame.")

        self.nodes = []
        self.node_list = []
        self.model = model
        self.model_name = model.__class__.__name__
        self.model_type = self.which_model()
        self.which_tree = None
        self.which_iteration = 0
        self.feature_names = feature_names
        self.target_names = target_names
        if feature_names is None:
            if isinstance(feature_data, pd.DataFrame):
                self.feature_names = feature_data.columns.tolist()
            elif (
                isinstance(feature_data, (list, np.ndarray))
                and len(feature_data) > 0
                and isinstance(feature_data[0], (list, np.ndarray))
            ):
                self.feature_names = [
                    f"feature{i}" for i in range(len(feature_data[0]))
                ]
            else:
                self.feature_names = []
        else:
            self.feature_names = feature_names

        if target_names is None:
            if isinstance(target_data, pd.DataFrame):
                self.target_names = target_data.columns.tolist() 
            elif isinstance(target_data, pd.Series):
                if self.model_type == "classification":
                    unique_values = target_data.unique()
                    if np.issubdtype(unique_values.dtype, np.integer):
                        self.target_names = [
                         f"target{i+1}" for i in range(len(unique_values))
                        ]
                    else:
                        self.target_names = unique_values.tolist()
                else:
                    self.target_names = "target_names"
            elif isinstance(target_data, (list, np.ndarray)):
                if self.model_type == "classification":
                    unique_values = np.unique(target_data)
                    if np.issubdtype(unique_values.dtype, np.integer):
                        self.target_names = [
                            f"target{i+1}" for i in range(len(unique_values))
                        ]
                    else:
                        self.target_names = unique_values.tolist()
                elif self.model_type == "regression":
                    self.target_names = ["target"]
                else:
                    self.target_names = "target"
            else:
                self.target_names = ["target"]
        else:
            self.target_names = target_names

        if isinstance(self.target_names, str):
            self.target_names = [self.target_names]

        self.feature_data = feature_data
        self.target_data = target_data
        if (self.feature_data is None and self.target_data is None):
            self.feature_data = [0]
            self.target_data = [0]
            self.model_type = "nodata"
            pass
        elif (feature_data is not None and self.target_data is not None):
            pass
        else:
            raise ValueError(
                "Target Data and Feature data should both exist or not")

        if isinstance(self.feature_data, pd.DataFrame):
            self.feature_data = self.feature_data.values
        if isinstance(self.target_data, pd.DataFrame):
            self.target_data = self.target_data.values

        self.target_len = len(self.target_names)
        self.tree_data = TreeData(
            self.model_type,
            self.feature_names,
            self.target_names,
            self.feature_data,
            self.target_data,
        )

        if self.which_model == "classification" and len(self.target_names) != np.unique(self.target_data):
            raise TypeError(
                "Invalid target_names length"
            )
        if self.which_model == "regression" and len(self.target_names) != 1:
            raise TypeError(
                "Invalid target_names length"
            )

        self.licence_key = licence_key

    def show_tree(self, which_tree=0):
        """
        Displaying model HTMl template and create json tree model.
        """
        if not isinstance(which_tree, int):
            raise TypeError("Invalid which_tree type. Expected an integer.")

        self.which_tree = which_tree

        if self.model_type == "uknown_model":
            return 0

        display(
            HTML(
                """
    <script src="https://cdn.jsdelivr.net/npm/d3@7" charset="utf-8"></script>
    <script src="https://cdn.jsdelivr.net/npm/tweetnacl@1.0.3/nacl.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/tweetnacl-util@0.15.1/nacl-util.min.js"></script>
"""
            )
        )

        combined_data_str = self.get_combined_data()

        display(HTML(templatehtml.get_d3_html(
            combined_data_str, self.licence_key)))

    def save_html(self, filename="output", which_tree=0):
        """
        Saving HTML file and create json tree model.
        """
        if not filename.endswith(".html"):
                filename += ".html"

        if not isinstance(which_tree, int):
            raise TypeError("Invalid which_tree type. Expected an integer.")

        if filename is not None and not isinstance(filename, str):
            raise TypeError("Invalid filename type. Expected a string.")

        self.which_tree = which_tree

        d3script = """
    <script src="https://cdn.jsdelivr.net/npm/d3@7" charset="utf-8"></script>
    <script src="https://cdn.jsdelivr.net/npm/tweetnacl@1.0.3/nacl.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/tweetnacl-util@0.15.1/nacl-util.min.js"></script>
"""

        combined_data_str = self.get_combined_data()

        html = d3script + \
            templatehtml.get_d3_html(combined_data_str, self.licence_key)

        with open(filename, "w", encoding="utf-8") as file:
            file.write(html)

        print(f"HTML saved to {filename}")

    def get_json_tree_data(self):
        """
        Save Tree Data to Json
        """
        tree_data_dict = self.tree_data.to_dict()
        tree_data_json = json.dumps(tree_data_dict, indent=4)
        return tree_data_json

    def get_json_tree(self):
        """
        Save Node Data to Json
        """
        for node_info in list(self.node_list):
            node = Node(
                node_info["feature"],
                round(node_info["threshold"], 3),
                round(node_info["impurity"], 3),
                node_info["samples"],
                node_info["class_distribution"],
                node_info["predicted_class"],
                node_info["is_leaf"],
                node_info["left_child_index"],
                node_info["right_child_index"],
            )
            self.nodes.append(node)

        self.create_node_dfs(0, "ROOT", None, None, None)
        root = self.nodes[0]
        if root.class_distribution is None and (self.model_type == "classification" or 
            (self.model_name=="GradientBoostingClassifier" and self.model_type == "nodata" )):
            self.count_class_distribution(root)
        tree_dict = root.to_dict()
        tree_json = json.dumps(tree_dict, indent=4)
        return tree_json

    def create_node_dfs(self, node_index, left_right, threshold, feature, x_axis):
        """
        Using DFS algorithm to create tree structure;
        """
        node = self.nodes[node_index]
        if left_right == "ROOT":
            node.start_end_x_axis = []
            for i in range(self.tree_data.feature_names_size):
                node.start_end_x_axis.append(["notexist", "notexist"])
        else:
            node.start_end_x_axis = deepcopy(x_axis)
        if (self.model_type != "nodata"):
            if left_right == "R" and feature >= 0:
                node.start_end_x_axis[feature][1] = threshold

            if left_right == "L" and feature >= 0:
                node.start_end_x_axis[feature][0] = threshold

        if node.left_children != -1:
            node.add_left(self.nodes[node.left_children])
            self.create_node_dfs(
                node.left_children,
                "L",
                node.threshold,
                node.feature,
                node.start_end_x_axis,
            )

        if node.right_children != -1:
            node.add_right(self.nodes[node.right_children])
            self.create_node_dfs(
                node.right_children,
                "R",
                node.threshold,
                node.feature,
                node.start_end_x_axis,
            )

    def save_json_tree(self, filename="treedata", which_tree=0):
        """
        Save tree to json tree.
        """
        if not filename.endswith(".json"):
            filename += ".json"

        if not isinstance(which_tree, int):
            raise TypeError("Invalid which_tree type. Expected an integer.")

        if filename is not None and not isinstance(filename, str):
            raise TypeError("Invalid filename type. Expected a string.")

        self.which_tree = which_tree

        combined_data_str = self.get_combined_data()

        with open((filename), "w", encoding="utf-8") as file:
            file.write(combined_data_str)

        print(f"JSON data saved to {filename}")

    def which_model(self):
        if self.model_name in (
            "DecisionTreeClassifier",
            "ExtraTreeClassifier",
            "ExtraTreesClassifier",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "LGBMClassifier",
            "XGBClassifier",
            "XGBRFClassifier",
        ):
            return "classification"
        elif self.model_name in (
            "DecisionTreeRegressor",
            "ExtraTreeRegressor",
            "ExtraTreesRegressor",
            "RandomForestRegressor",
            "GradientBoostingRegressor",
            "LGBMRegressor",
            "XGBRegressor",
            "XGBRFRegressor",
        ):
            return "regression"

        elif self.model_name in ("ModelLoader"):
            if self.model.model_type == "classification":
                return "classification"
            elif self.model.model_type == "regression":
                return "regression"
            else:
                print("Uknown Model")
                return "uknown_model"

        elif self.model_name in ("Booster"):
            if hasattr(self.model, "get_dump"):
                self.model_name = "XGBoostBooster"
                config_str = self.model.save_config()
                config = json.loads(config_str)
                objective = (
                    config.get("learner", {}).get(
                        "objective", {}).get("name", "")
                )
                if "binary:logistic" in objective or "multi:" in objective:
                    return "classification"
                elif "reg:" in objective:
                    return "regression"
                else:
                    print("Uknown model type")
            elif hasattr(self.model, "dump_model"):
                self.model_name = "LightGBMBooster"
                model_dict = self.model.dump_model()
                if model_dict["objective"] == "regression":
                    return "regression"
                else:
                    return "classification"
        else:
            print("Uknown Model")
            return "uknown_model"

    def convert_model_to_dict_array(self):
        model_name = self.model_name
        if model_name in (
            "DecisionTreeClassifier",
            "ExtraTreeClassifier",
            "ExtraTreesClassifier",
            "DecisionTreeRegressor",
            "ExtraTreeRegressor",
            "ExtraTreesRegressor",
            "RandomForestClassifier",
            "RandomForestRegressor",
            "GradientBoostingRegressor",
            "GradientBoostingClassifier",
        ):
            if model_name in (
                "RandomForestClassifier",
                "RandomForestRegressor",
                "ExtraTreesClassifier",
                "ExtraTreesRegressor",
            ):
                if 0 <= self.which_tree < len(self.model.estimators_):
                    super_tree = self.model.estimators_[self.which_tree].tree_
                else:
                    raise IndexError("Out of trees range")
            elif model_name in (
                "GradientBoostingRegressor",
                "GradientBoostingClassifier",
            ):
                if (0 <= self.which_tree < len(self.model.estimators_) and 0 <= self.which_iteration < len(self.model.estimators_[self.which_tree]) ):
                    super_tree = self.model.estimators_[self.which_tree, self.which_iteration].tree_
                else:
                    raise IndexError("Wartość 'which_tree' lub 'which_iteration' jest poza zakresem dostępnych wartości.")
            else:
                super_tree = self.model.tree_

            
            sklearn_version = importlib.metadata.version('scikit-learn')

            for i in range(super_tree.node_count):
                samples = super_tree.n_node_samples[i]
                if(self.model_name not in ("GradientBoostingClassifier")):
                    if(sklearn_version > "1.4.0"):
                        class_distribution = super_tree.value[i] * samples
                    else:
                        class_distribution = super_tree.value[i]

                    predicted_class_index = class_distribution.argmax()
                else:
                    class_distribution = None
                    predicted_class_index = "brak"
                is_leaf = False
                if self.model_type != "nodata" and self.model_name not in ("GradientBoostingClassifier"):
                    predicted_class = self.target_names[predicted_class_index]
                else:
                    if(self.model_name == "GradientBoostingClassifier"):
                        predicted_class = "No data"
                    else:
                        predicted_class = predicted_class_index
                left_children = super_tree.children_left[i]
                right_children = super_tree.children_right[i]
                if right_children == -1 and left_children == -1:
                    is_leaf = True

                node_info = {
                    "index": i,
                    "feature": super_tree.feature[i],
                    "impurity": super_tree.impurity[i],
                    "threshold": super_tree.threshold[i],
                    "class_distribution": class_distribution,
                    "predicted_class": predicted_class,
                    "samples": samples,
                    "is_leaf": is_leaf,
                    "left_child_index": left_children,
                    "right_child_index": right_children,
                }
                self.node_list.append(node_info)
        if model_name == "LightGBMBooster" or model_name in ("LGBMRegressor", "LGBMClassifier"):
            if model_name != "LightGBMBooster":
                booster = self.model.booster_
                model_dict = booster.dump_model()
            else:
                model_dict = self.model.dump_model()

            if 0 <= self.which_tree < len(model_dict["tree_info"]):
                tree = model_dict["tree_info"][self.which_tree]
            else:
                raise IndexError("Tree index out of range")
            tree = model_dict["tree_info"][self.which_tree]
            self.collect_node_info_lgbm(tree["tree_structure"])
        if model_name in ("XGBoostBooster", "XGBClassifier", "XGBRegressor", "XGBRFClassifier", "XGBRFRegressor"):
            if model_name != "XGBoostBooster":
                booster = self.model.get_booster()
                model_dict = booster.get_dump(with_stats=True, dump_format="json")
            else:
                model_dict = self.model.get_dump(with_stats=True, dump_format="json")


            json_tree = model_dict[self.which_tree]

            if 0 <= self.which_tree < len(model_dict):
                json_tree = model_dict[self.which_tree]
            else:
                raise KeyError(f"Wartość 'which_tree' ({self.which_tree}) nie jest poprawnym kluczem w 'model_dict'.")
            dict_tree = json.loads(json_tree)
            self.collect_node_info_xgboost(dict_tree)
        if model_name in ("ModelLoader"):
            self.node_list  = self.model.model_dict


    def collect_node_info_lgbm(self, node, depth=0):
        node_index = len(self.node_list)
        if "split_index" in node:
            predicted_data = None
            if self.model_type == "nodata":
                predicted_data = "No data"
            else:
                predicted_data = self.target_names[0]

            class_dist = [[10, 10, 10]]
            if self.model_type == "classification":
                class_dist = None
            node_info = {
                "index": node_index,
                "feature": node["split_feature"],
                "impurity": node["split_gain"],
                "threshold": node["threshold"],
                "class_distribution": class_dist,
                "predicted_class": predicted_data,
                "samples": node["internal_count"],
                "is_leaf": False,
                "left_child_index": None,
                "right_child_index": None,
            }
            self.node_list.append(node_info)
            left_child_index = self.collect_node_info_lgbm(
                node["left_child"], depth + 1
            )
            right_child_index = self.collect_node_info_lgbm(
                node["right_child"], depth + 1
            )
            self.node_list[node_index]["left_child_index"] = left_child_index
            self.node_list[node_index]["right_child_index"] = right_child_index
        else:
            predicted_data = None
            if self.model_type == "nodata":
                predicted_data = "No data"
            else:
                predicted_data = self.target_names[0]
            class_dist = [[10, 10, 10]]
            if self.model_type == "classification":
                class_dist = None
            node_info = {
                "index": node_index,
                "feature": -1,
                "impurity": 0,
                "threshold": -1,
                "class_distribution": class_dist,
                "predicted_class": predicted_data,
                "samples": node["leaf_count"],
                "is_leaf": True,
                "left_child_index": -1,
                "right_child_index": -1,
            }

            self.node_list.append(node_info)

        return node_index

    def count_class_distribution(self, node):
        target_len = self.target_len
        node.class_distribution = [0] * target_len
        index_set = set()
        for i in range(len(node.start_end_x_axis)):
            for j in range(len(self.feature_data)):
                if node.start_end_x_axis[i][0] != "notexist":
                    if node.start_end_x_axis[i][0] < self.feature_data[j][i]:
                        index_set.add(j)

                if node.start_end_x_axis[i][1] != "notexist":
                    if node.start_end_x_axis[i][1] > self.feature_data[j][i]:
                        index_set.add(j)

        for i in range(len(self.target_data)):
            if i not in index_set:
                node.class_distribution[self.target_data[i]] += 1

        node.class_distribution = [node.class_distribution]

        if node.left_node != None:
            self.count_class_distribution(node.left_node)

        if node.right_node != None:
            self.count_class_distribution(node.right_node)

    def get_combined_data(self):
        self.convert_model_to_dict_array()
        json_node_data_str = self.get_json_tree()
        json_tree_data_str = self.get_json_tree_data()

        json_node_data = json.loads(json_node_data_str)
        json_tree_data = json.loads(json_tree_data_str)

        combined_data = {
            "node_data": json_node_data,
            "tree_data": json_tree_data,
        }
        combined_data_str = json.dumps(combined_data)

        return combined_data_str

    def collect_node_info_xgboost(self, node, depth=0):
        node_index = len(self.node_list)

        if "split" in node:
            feature_index = 0
            feature = node["split"]
            if feature.startswith("f") and len(feature) <= 3:
                feature_index = int(feature[1:])
            else:
                try:
                    feature_index = self.feature_names.index(feature)
                except ValueError:
                    raise ValueError(
                        f"Feature {feature} not found in feature_names.")

            class_dist = [[10, 10, 10]]
            predicted_data = None
            if self.model_type == "nodata":
                predicted_data = "No data"
            else:
                predicted_data = self.target_names[0]

            if self.model_type == "classification":
                class_dist = None
            node_info = {
                "index": node_index,
                "feature": int(feature_index),
                "impurity": node.get("gain", 0),
                "threshold": node["split_condition"],
                "class_distribution": class_dist,
                "predicted_class": predicted_data,
                "samples": node.get("cover", 0),
                "is_leaf": False,
                "left_child_index": None,
                "right_child_index": None,
            }
            self.node_list.append(node_info)
            left_child_index = self.collect_node_info_xgboost(
                node["children"][0], depth + 1
            )
            right_child_index = self.collect_node_info_xgboost(
                node["children"][1], depth + 1
            )
            self.node_list[node_index]["left_child_index"] = left_child_index
            self.node_list[node_index]["right_child_index"] = right_child_index
        else:
            class_dist = [[10, 10, 10]]
            if self.model_type == "classification":
                class_dist = None

            predicted_data = None
            if self.model_type == "nodata":
                predicted_data = "No data"
            else:
                predicted_data = self.target_names[0]
            node_info = {
                "index": node_index,
                "feature": -1,
                "impurity": 0,
                "threshold": -1,
                "class_distribution": class_dist,
                "predicted_class": predicted_data,
                "samples": node.get("cover", 0),
                "is_leaf": True,
                "left_child_index": -1,
                "right_child_index": -1,
            }
            self.node_list.append(node_info)

        return node_index
