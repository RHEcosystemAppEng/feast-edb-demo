from feast import Entity, ValueType

__all__ = ['ENTITY_NAME_USER', 'ENTITY_NAME_ITEM']

user_id = 'user_id'
item_id = 'item_id'
ENTITY_NAME_USER = user_id.split('_')[0]
ENTITY_NAME_ITEM = item_id.split('_')[0]

user = Entity(
    name=ENTITY_NAME_USER,
    join_keys=[user_id],
    value_type=ValueType.INT64, #TODO make sure for type
    description=user_id.replace('_', ''),
)

item = Entity(
    name=ENTITY_NAME_ITEM,
    join_keys=[item_id],
    value_type=ValueType.INT64, #TODO make sure for type
    description=item_id.replace('_', ''),
)
