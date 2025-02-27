from feast import FeatureService

from feature_repo.feature_views import user_feature_view, item_feature_view, interaction_feature_view

feature_service = FeatureService(
    name="model_v1",
    features=[user_feature_view, item_feature_view, interaction_feature_view],
)
feature_service2 = FeatureService(
    name="model_v2",
    features=[user_feature_view, item_feature_view, interaction_feature_view],
)
