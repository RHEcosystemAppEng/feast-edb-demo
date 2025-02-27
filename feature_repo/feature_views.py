from datetime import timedelta

from feast import (
    FeatureView,
    Field,
)
from feast.types import Float32

from data_sources import interactions_source, items_source, users_source
from entities import ENTITY_NAME_ITEM, ENTITY_NAME_USER
from feast.types import Float32, Int32, Int64, String, Bool


# Define feature views
user_feature_view = FeatureView(
    name="user_features",
    entities=[ENTITY_NAME_USER],
    ttl=timedelta(days=365 * 2),  # Features valid for 2 years
    schema=[
        Field(name="user_id", dtype=Int64),
        Field(name="age", dtype=Int32),
        Field(name="gender", dtype=String),
        Field(name="preferences", dtype=String),
    ],
    source=users_source,
    online=False
)

item_feature_view = FeatureView(
    name="item_features",
    entities=[ENTITY_NAME_ITEM],
    ttl=timedelta(days=365),  # Features valid for 1 year
    schema=[
        Field(name="item_id", dtype=Int64),
        Field(name="category", dtype=String),
        Field(name="subcategory", dtype=String),
        Field(name="price", dtype=Float32),
        Field(name="avg_rating", dtype=Float32),
        Field(name="num_ratings", dtype=Int32),
        Field(name="popular", dtype=Bool),
        Field(name="new_arrival", dtype=Bool),
        Field(name="on_sale", dtype=Bool),
    ],
    source=items_source,
    online=False
)

# User-item interaction feature view
interaction_feature_view = FeatureView(
    name="interactions_features",
    entities=[ENTITY_NAME_USER, ENTITY_NAME_ITEM],
    ttl=timedelta(days=90),  # Recent interactions are more relevant
    schema=[
        Field(name="user_id", dtype=Int64),
        Field(name="item_id", dtype=Int64),
        Field(name="interaction_type", dtype=String),
        Field(name="rating", dtype=Int32),
        Field(name="quantity", dtype=Int32),
    ],
    source=interactions_source,
    online=False
)



# # A push source is useful if you have upstream systems that transform features (e.g. stream processing jobs)
# driver_stats_push_source = PushSource(
#     name="driver_stats_push_source", batch_source=driver_stats,
# )


# user_view = FeatureView(
#     name=USER_VIEW_NAME,
#     description=USER_VIEW_DESCRIPTION,
#     entities=[ENTITY_NAME_USER],
#     ttl=timedelta(weeks=2),
#     schema=[
#         Field(name=FEATURE_1_USER, dtype=Float32), # TODO change dtype
#         Field(name=FEATURE_2_USER, dtype=Float32), # TODO change dtype
#     ],
#     online=False,
#     source=users_source,
#     # tags={"production": "True"}
# )

# item_view = FeatureView(
#     name=ITEM_VIEW_NAME,
#     description=ITEM_VIEW_DESCRIPTION,
#     entities=[ENTITY_NAME_ITEM],
#     ttl=timedelta(weeks=2),
#     schema=[
#         Field(name=FEATURE_1_ITEM, dtype=Float32), # TODO change dtype
#         Field(name=FEATURE_2_ITEM, dtype=Float32), # TODO change dtype
#     ],
#     online=False,
#     source=SOURCE_NAME,
#     # tags={"production": "True"}
# )
