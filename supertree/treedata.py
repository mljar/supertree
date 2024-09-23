import numpy as np
import pandas as pd


class TreeData:

    def __init__(
        self, tree_type, feature_names, target_names, data_feature, data_target, model_name
    ):
        self.feature_names = feature_names
        self.target_names = None
        if  isinstance(self.target_names,str):
            self.target_names = [target_names]
        else:
            self.target_names = target_names
        self.data_feature = self.extract_values_if_dataframe(data_feature)
        self.data_target = self.extract_values_if_dataframe(data_target)
        self.data_target = self.convert_target_strings(data_target)
        self.feature_names_size = len(feature_names)
        self.tree_type = tree_type
        self.max_samples = None
        self.show_sample = "nodata"
        self.model_name = model_name
        self.which_tree = 0

    def to_dict(self):
        def convert(value):
            if isinstance(
                value,
                (
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                ),
            ):
                return int(value)
            if isinstance(value, (np.float16, np.float32, np.float64) + (getattr(np, 'float128', ()),)):
                return float(value)
            if isinstance(value, np.ndarray):
                return [convert(item) for item in value.tolist()]
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, pd.Series):
                return [convert(item) for item in value.tolist()]
            if isinstance(value, dict):
                return {key: convert(val) for key, val in value.items()}

            return value

        data_feature = self.data_feature[:self.max_samples] if len(self.data_feature) > self.max_samples else self.data_feature
        data_target = self.data_target[:self.max_samples] if len(self.data_target) > self.max_samples else self.data_target

        tree_data_dict = {
            "tree_type": self.tree_type,
            "feature_names": self.feature_names,
            "target_names": convert(self.target_names),
            "data_feature": convert(data_feature),
            "data_target": convert(data_target),
            "show_sample": convert(self.show_sample),
            "model_name": convert(self.model_name),
            "which_tree": convert(self.which_tree),
        }

        return tree_data_dict

    def extract_values_if_dataframe(self, data):
        """
        Conver dataframe.
        """
        if isinstance(data, pd.DataFrame):
            return data.values
        return data
        

    def convert_target_strings(self, data_target):
        """
        convert_strings
        """
        if isinstance(data_target[0], str):
            target_map = {name: idx for idx, name in enumerate(self.target_names)}
            data_target = [target_map[val] for val in data_target]
        if all(1 <= val <= len(self.target_names) for val in data_target):
            data_target = [val - 1 for val in data_target]

        return data_target

    def set_show_sample(self, sample):
        """
        Set and check show sample
        """
        if not isinstance(sample, list):
            raise TypeError(f"Expected a list, but got {type(sample).__name__}")

        if len(sample) != self.feature_names_size:
            raise ValueError(f"Expected a list of length {self.feature_names_size}, but got {len(sample)}")

        self.show_sample = sample

    def reset_sample(self):
        self.show_sample = "nodata"

    def set_which_tree(self,which):
        self.which_tree = which

