from typing import Dict, List, Union

def flatten_entities(entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Filters out empty lists and blank strings.
    """
    
    if not entities:
        return {}

    cleaned: Dict[str, List[str]] = {}
    for category, values in entities.items():
        if not isinstance(values, list):
            continue
        filtered = [v.strip() for v in values if isinstance(v, str) and v.strip()]
        if filtered:
            cleaned[category] = filtered

    return cleaned


def flatten_entities_to_list(entities: Dict[str, List[str]]) -> List[str]:
    """
    Flatten to a plain list of entity name strings.
    """
    result = []
    for values in entities.values():
        if isinstance(values, list):
            result.extend(v.strip() for v in values if isinstance(v, str) and v.strip())
    return result