# Adapted from https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/utils/_merge.py
# Also tweaked to more closely match https://github.com/hadley/ellmer/blob/main/R/utils-merge.R

from __future__ import annotations

from typing import Any, Optional


def merge_dicts(left: dict[str, Any], *others: dict[str, Any]) -> dict[str, Any]:
    """Merge many dicts, handling specific scenarios where a key exists in both
    dictionaries but has a value of None in 'left'. In such cases, the method uses the
    value from 'right' for that key in the merged dictionary.

    Args:
        left: The first dictionary to merge.
        others: The other dictionaries to merge.

    Returns:
        The merged dictionary.

    Raises:
        TypeError: If the key exists in both dictionaries but has a different type.
        TypeError: If the value has an unsupported type.

    Example:
        If left = {"function_call": {"arguments": None}} and
        right = {"function_call": {"arguments": "{\n"}}
        then, after merging, for the key "function_call",
        the value from 'right' is used,
        resulting in merged = {"function_call": {"arguments": "{\n"}}.
    """
    merged = left.copy()
    for right in others:
        for right_k, right_v in right.items():
            left_v = merged.get(right_k, None)

            if right_v is None:
                if right_k not in merged:
                    merged[right_k] = None
            elif left_v is None:
                merged[right_k] = right_v
            elif left_v == right_v:
                continue
            elif isinstance(left_v, str):
                merged[right_k] += right_v
            elif isinstance(left_v, (int, float)):
                merged[right_k] = right_v
            elif isinstance(merged[right_k], dict):
                merged[right_k] = merge_dicts(merged[right_k], right_v)
            elif isinstance(merged[right_k], list):
                merged[right_k] = merge_lists(merged[right_k], right_v)
            elif type(merged[right_k]) is not type(right_v):
                raise TypeError(
                    f'additional_kwargs["{right_k}"] already exists in this message,'
                    " but with a different type."
                )
            else:
                raise TypeError(
                    f"Additional kwargs key {right_k} already exists in left dict and "
                    f"value has unsupported type {type(merged[right_k])}."
                )
    return merged


def merge_lists(
    left: Optional[list[Any]], *others: Optional[list[Any]]
) -> Optional[list[Any]]:
    """Add many lists, handling None.

    Args:
        left: The first list to merge.
        others: The other lists to merge.

    Returns:
        The merged list.
    """
    merged = left.copy() if left is not None else None
    for other in others:
        if other is None:
            continue
        elif merged is None:
            merged = other.copy()
        else:
            for e in other:
                if isinstance(e, dict) and "index" in e and isinstance(e["index"], int):
                    to_merge = [
                        i
                        for i, e_left in enumerate(merged)
                        if e_left["index"] == e["index"]
                    ]
                    if to_merge:
                        # TODO: Remove this once merge_dict is updated with special
                        # handling for 'type'.
                        if "type" in e:
                            e: dict[str, Any] = {  # noqa: PLW2901
                                k: v for k, v in e.items() if k != "type"
                            }
                        merged[to_merge[0]] = merge_dicts(merged[to_merge[0]], e)
                    else:
                        merged.append(e)
                else:
                    merged.append(e)
    return merged
