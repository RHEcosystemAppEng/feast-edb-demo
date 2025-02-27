from feast import FileSource
from feast.data_format import ParquetFormat
import os

data_path = 'data'

users_source = FileSource(
    file_format=ParquetFormat(),
    path=os.path.join(data_path, 'recommendation_users.parquet'),
    # timestamp_field="signup_date", # I feel like this useless
)
interactions_source = FileSource(
    file_format=ParquetFormat(),
    path=os.path.join(data_path, 'recommendation_interactions.parquet'),
    timestamp_field="timestamp",
)
items_source = FileSource(
    file_format=ParquetFormat(),
    path=os.path.join(data_path, 'recommendation_items.parquet'),
    # timestamp_field="event_timestamp",
)