from feast import FileSource, PushSource
from feast.data_format import ParquetFormat
import os

feast_path = 'feature_repo'
data_path = 'data'

users_source = FileSource(
    file_format=ParquetFormat(),
    path=os.path.join(data_path, 'recommendation_users.parquet'),
    timestamp_field="signup_date",
)
interactions_source = FileSource(
    file_format=ParquetFormat(),
    path=os.path.join(data_path, 'recommendation_interactions.parquet'),
    timestamp_field="timestamp",
)
items_source = FileSource(
    file_format=ParquetFormat(),
    path=os.path.join(data_path, 'recommendation_items.parquet'),
    timestamp_field="arrival_date",
)

item_embed_push_source = PushSource(
    name='item_embed_push_source',
    batch_source=items_source
)

user_embed_push_source = PushSource(
    name='user_embed_push_source',
    batch_source=users_source    
)