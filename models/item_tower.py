import torch
import torch.nn as nn

class ItemTower(nn.Module):
    def __init__(self, d_model=64, category_count=5, subcategory_count=25):
        super().__init__()
        # Define ratios for each feature.
        ratios = {
            'category': 3, 'subcategory': 3,
            'price': 3, 'avg_rating': 2, 'num_ratings': 2,
            'popular': 1, 'new_arrival': 1, 'on_sale': 1, 'arrival_date': 1
        }
        total_ratio = sum(ratios.values())
        unit_dim = d_model // total_ratio
        dim_reminder = d_model % total_ratio

        # Define the dimention
        dims = {k: v * unit_dim for k, v in ratios.items()}
        # add the leftover dim to subcategory
        dims['subcategory'] = dims.get('subcategory') + dim_reminder

        # Create embedding modules for categorical features.
        self.embeds = nn.ModuleDict({
            'category': nn.Embedding(category_count, dims['category']),
            'subcategory': nn.Embedding(subcategory_count, dims['subcategory'])
        })

        # Define keys for numeric features.
        self.numeric_keys = ['price', 'avg_rating', 'num_ratings', 'popular', 'new_arrival', 'on_sale', 'arrival_date']
        
        # Create projection and normalization modules for each numeric feature.
        self.encoders = nn.ModuleDict({k: nn.Linear(1, dims[k]) for k in self.numeric_keys})
        self.norms = nn.ModuleDict({k: nn.RMSNorm(dims[k]) for k in self.numeric_keys})

    def forward(self,
                category, subcategory,
                price, avg_rating, num_ratings,
                popular, new_arrival, on_sale, arrival_date):
        # Process categorical features.
        cat_out = [
            self.embeds['category'](category),
            self.embeds['subcategory'](subcategory)
        ]
        # Process numeric features using a loop.
        num_inputs = [price, avg_rating, num_ratings, popular, new_arrival, on_sale, arrival_date]
        num_out = [self.norms[k](self.encoders[k](x)) for k, x in zip(self.numeric_keys, num_inputs)]
        # Concatenate all feature representations.
        return torch.cat(cat_out + num_out, dim=-1)
