
class ModelLoader():
    def __init__(self,model_type, model_dict_list):
        self.model_type = model_type
        self.model_dict = model_dict_list

        required_keys = {"index", "feature", "impurity", "threshold", 
                         "class_distribution", "predicted_class", 
                         "samples", "is_leaf", "left_child_index", "right_child_index"}
        
        # Validate model_dict_list
        if not isinstance(model_dict_list, list):
            raise TypeError("model_dict_list must be a list of dictionaries.")
        
        for item in model_dict_list:
            if not isinstance(item, dict):
                raise TypeError("Each item in model_dict_list must be a dictionary.")
            
            if not required_keys.issubset(item.keys()):
                missing_keys = required_keys - item.keys()
                raise ValueError(f"Dictionary is missing required keys: {missing_keys}")
        
        self.model_dict = model_dict_list
