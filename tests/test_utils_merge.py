from chatlas._merge import merge_dicts


def test_nulls_and_values():
    assert merge_dicts({}, {"a": 1}) == {"a": 1}
    assert merge_dicts({"a": 1}, {}) == {"a": 1}


def test_equal_values():
    assert merge_dicts({"a": 1}, {"a": 1}) == {"a": 1}
    assert merge_dicts({"a": 1.5}, {"a": 1.5}) == {"a": 1.5}
    assert merge_dicts({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}

    assert merge_dicts({"a": True}, {"a": True}) == {"a": True}
    assert merge_dicts({"a": False}, {"a": False}) == {"a": False}

    assert merge_dicts({"a": "x"}, {"a": "x"}) == {"a": "x"}


def test_strings_are_concatenated():
    assert merge_dicts({"a": "a"}, {"a": "b"}) == {"a": "ab"}
    assert merge_dicts({"a": {"b": "a"}}, {"a": {"b": "b"}}) == {"a": {"b": "ab"}}


def test_merge_dictionaries_with_different_keys():
    assert merge_dicts({"a": 1, "b": 2}, {"a": 1}) == {"a": 1, "b": 2}

    assert merge_dicts({"a": 1, "b": 2}, {"c": None}) == {"a": 1, "b": 2, "c": None}


def test_nulls_dont_overwrite_existing_values():
    assert merge_dicts({"a": 1, "b": 2}, {"a": None}) == {"a": 1, "b": 2}


def test_can_merge_lists():
    assert merge_dicts({"a": [1, 2]}, {}) == {"a": [1, 2]}

    assert merge_dicts({}, {"a": [1, 2]}) == {"a": [1, 2]}

    assert merge_dicts({"a": [1, 2]}, {"a": [3]}) == {"a": [1, 2, 3]}


def test_respects_index_when_merging_lists():
    assert merge_dicts(
        {"a": [{"index": 0, "b": "{"}]}, {"a": [{"index": 0, "b": "f"}]}
    ) == {"a": [{"index": 0, "b": "{f"}]}

    assert merge_dicts(
        {"a": [{"index": 0, "b": "a"}]}, {"a": [{"index": 1, "b": "b"}]}
    ) == {"a": [{"index": 0, "b": "a"}, {"index": 1, "b": "b"}]}
