from models.two_tower import TwoTowerModel
import pandas as pd
from models.data_util import data_preproccess
from models.user_tower import UserTower
from models.item_tower import ItemTower


def train_two_tower(item_tower: ItemTower, user_tower: UserTower, item_df: pd.DataFrame, user_df: pd.DataFrame, interaction_df: pd.DataFrame, neg_interaction_df: pd.DataFrame):
    categories_values = {
        'preferences': ['Electronics', 'Books', 'Clothing', 'Home', 'Sports'],
        'category': ['Electronics', 'Books', 'Clothing', 'Home', 'Sports'],
        'subcategory': ['Smartphones', 'Laptops', 'Cameras', 'Audio', 'Accessories', 'Fiction', 'Non-fiction', 'Science', 'History', 'Self-help', 'Clothing', 'Shirts', 'Pants', 'Dresses', 'Shoes', 'Accessories', 'Home', 'Kitchen', 'Furniture', 'Decor', 'Bedding', 'Appliances', 'Sports', 'Fitness', 'Outdoor', 'Team Sports', 'Footwear', 'Equipment'],
        'interaction_type': ['view', 'cart', 'purchase', 'rate']
    }
    

    #TODO implement me if needed
    proccesed_items_dict = data_preproccess(item_df, categories_values)
    proccesed_users_dict = data_preproccess(user_df, categories_values)
    proccesed_real_intercations = data_preproccess(interaction_df, category_values=categories_values)
    proccesed_neg_intercations = data_preproccess(neg_interaction_df, category_values=categories_values)
    
    two_tower_model = TwoTowerModel(item_tower, user_tower)
    two_tower_model(proccesed_items_dict, proccesed_users_dict, proccesed_real_intercations, proccesed_neg_intercations)
    
