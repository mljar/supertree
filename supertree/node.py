import numpy as np


class Node:
    def __init__(self, feature, threshold, impurity, samples,
                 class_distribution,
                 treeclass, is_leaf,
                 left_children, right_children):
        self.feature = feature
        self.threshold = threshold
        self.impurity = impurity
        self.samples = samples
        self.class_distribution = class_distribution
        self.treeclass = treeclass
        self.is_leaf = is_leaf
        self.left_children = left_children
        self.right_children = right_children
        self.left_node = None
        self.right_node = None
        self.start_end_x_axis = []


    def to_dict(self):
        def convert(value):
            if isinstance(value, (np.int64, np.int32)):
                return int(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            return value

        node_dict = {
            "feature": int(self.feature) if isinstance(self.feature, np.longlong) else self.feature,
            "threshold": self.threshold,
            "impurity": self.impurity,
            "samples": int(self.samples) if isinstance(self.samples, np.longlong) else self.samples,
            "class_distribution": [convert(val) for val in self.class_distribution],
            "treeclass": convert(self.treeclass),
            "is_leaf": self.is_leaf,
            "start_end_x_axis": [convert(val) for val in self.start_end_x_axis] if self.start_end_x_axis is not None else None,

        }

        if self.left_node is not None:
            node_dict["left_node"] = self.left_node.to_dict()
        else:
            node_dict["left_node"] = None

        if self.right_node is not None:
            node_dict["right_node"] = self.right_node.to_dict()
        else:
            node_dict["right_node"] = None

        return node_dict
    

    def add_left(self, left_node):
        self.left_node = left_node

    def add_right(self, right_node):
        self.right_node = right_node

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.__str__()