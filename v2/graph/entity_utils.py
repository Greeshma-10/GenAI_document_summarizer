def flatten_entities(entity_dict):

    entities = []

    for category in entity_dict.values():
        entities.extend(category)

    # remove duplicates
    entities = list(set(entities))

    return entities