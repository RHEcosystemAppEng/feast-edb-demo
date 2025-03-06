from feast import FeatureService

from feature_views import user_feature_view, item_feature_view, interaction_feature_view, item_embedding_view

feature_service = FeatureService(
    name="model_v1",
    features=[user_feature_view, item_feature_view, interaction_feature_view],
)
feature_service2 = FeatureService(
    name="model_v2",
    features=[user_feature_view, item_feature_view, interaction_feature_view],
)
item_feature_service = FeatureService(
    name="item_service",
    features=[item_feature_view]
)
user_feature_service = FeatureService(
    name="user_service",
    features=[user_feature_view]
)
interactions_feature_service = FeatureService(
    name="interaction_service",
    features=[interaction_feature_view]
)

item_embedding_service = FeatureService(
    name='item_embedding',
    features=[item_embedding_view]
)