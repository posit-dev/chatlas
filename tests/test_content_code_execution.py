from chatlas._content import (
    ContentToolRequestCodeExecution,
    ContentToolResponseCodeExecution,
    ContentUnion,
    create_content,
)


def test_content_tool_request_code_execution_defaults():
    content = ContentToolRequestCodeExecution(code="print(1 + 1)")
    assert content.content_type == "code_execution_request"
    assert content.code == "print(1 + 1)"
    assert content.language is None
    assert content.extra is None


def test_content_tool_request_code_execution_str():
    content = ContentToolRequestCodeExecution(code="print(1 + 1)")
    assert "print(1 + 1)" in str(content)


def test_content_tool_response_code_execution_defaults():
    content = ContentToolResponseCodeExecution(output="2")
    assert content.content_type == "code_execution_result"
    assert content.output == "2"
    assert content.error is None
    assert content.container_id is None
    assert content.extra is None


def test_content_tool_response_code_execution_str_with_output():
    content = ContentToolResponseCodeExecution(output="2")
    assert "2" in str(content)


def test_content_tool_response_code_execution_str_with_error():
    content = ContentToolResponseCodeExecution(error="NameError: x is not defined")
    assert "NameError" in str(content)


def test_create_content_round_trips_request():
    content = ContentToolRequestCodeExecution(
        code="print(1 + 1)", language="PYTHON", extra={"id": "abc"}
    )
    data = content.model_dump()
    restored = create_content(data)
    assert isinstance(restored, ContentToolRequestCodeExecution)
    assert restored.code == "print(1 + 1)"
    assert restored.language == "PYTHON"
    assert restored.extra == {"id": "abc"}


def test_create_content_round_trips_response():
    content = ContentToolResponseCodeExecution(
        output="2", container_id="cntr_123", extra={"id": "abc"}
    )
    data = content.model_dump()
    restored = create_content(data)
    assert isinstance(restored, ContentToolResponseCodeExecution)
    assert restored.output == "2"
    assert restored.container_id == "cntr_123"


def test_content_union_includes_code_execution_types():
    import typing

    args = typing.get_args(ContentUnion)
    assert ContentToolRequestCodeExecution in args
    assert ContentToolResponseCodeExecution in args
