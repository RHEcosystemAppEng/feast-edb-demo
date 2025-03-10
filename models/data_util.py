import pandas as pd
import torch
from typing import Dict, List
from pandas.api.types import is_datetime64_any_dtype

def data_preproccess(df: pd.DataFrame, category_values: Dict[str, List[str]]):
    features = [feature for feature in df.columns if not feature.endswith('_id')]
    proccesed_tensor_dict = dict()
    for feature in features:
        # feature is category 
        if feature in category_values.keys():
            categories = category_values.get(feature)
            # map numbers to each category
            category_num = {category: i for category, i, in zip(categories, range(len(categories)))}
            x_feature = df[feature].map(category_num)
        # datetime case
        elif is_datetime64_any_dtype(df[feature]):
            x_feature = df[feature].apply(lambda x: x.toordinal())
        # numerical case
        else:
            x_feature = df[feature]
            
        # parse to tensor
        x_feature = torch.tensor(x_feature)
        proccesed_tensor_dict[feature] = x_feature
    return proccesed_tensor_dict
            
            