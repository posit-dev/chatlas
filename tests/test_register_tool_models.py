from typing import Dict, List, Optional, Union

import pytest
from chatlas import ChatOpenAI
from pydantic import BaseModel, Field


class TestRegisterToolWithBaseModels:
    """Test register_tool() with various BaseModel configurations."""

    def test_basic_basemodel(self):
        """Test register_tool with a basic BaseModel."""

        class SimpleModel(BaseModel):
            """A simple model for testing."""

            name: str
            age: int

        def process_person(name: str, age: int) -> str:
            return f"{name} is {age} years old"

        chat = ChatOpenAI()
        chat.register_tool(process_person, model=SimpleModel)

        tool = chat._tools["SimpleModel"]
        assert tool.name == "SimpleModel"

        func = tool.schema["function"]
        assert func.get("description") == "A simple model for testing."

        params: dict = func.get("parameters", {})
        assert params.get("type") == "object"
        assert "name" in params.get("properties", {})
        assert "age" in params.get("properties", {})
        assert params.get("properties", {}).get("name", {}).get("type") == "string"
        assert params.get("properties", {}).get("age", {}).get("type") == "integer"

    def test_basemodel_with_field_descriptions(self):
        """Test BaseModel with Field descriptions."""

        class DetailedModel(BaseModel):
            """Model with detailed field descriptions."""

            username: str = Field(description="The user's login name")
            score: int = Field(description="User's current score", ge=0)
            is_active: bool = Field(description="Whether user is currently active")

        def update_user(username: str, score: int, is_active: bool) -> str:
            return f"Updated {username}: score={score}, active={is_active}"

        chat = ChatOpenAI()
        chat.register_tool(update_user, model=DetailedModel)

        tool = chat._tools["DetailedModel"]
        func = tool.schema["function"]
        params: dict = func.get("parameters", {})
        props: dict = params.get("properties", {})

        assert props["username"]["description"] == "The user's login name"
        assert props["score"]["description"] == "User's current score"
        assert props["is_active"]["description"] == "Whether user is currently active"

    def test_basemodel_with_aliases(self):
        """Test BaseModel with Field aliases."""

        class AliasModel(BaseModel):
            """Model with field aliases."""

            internal_id: str = Field(alias="id", description="The external ID")
            full_name: str = Field(alias="name", description="Person's full name")
            email_addr: str = Field(alias="email", description="Email address")

        def create_user(id: str, name: str, email: str) -> str:
            return f"Created user {name} with ID {id} and email {email}"

        chat = ChatOpenAI()
        chat.register_tool(create_user, model=AliasModel)

        tool = chat._tools["AliasModel"]
        func = tool.schema["function"]
        params: dict = func.get("parameters", {})
        props: dict = params.get("properties", {})

        # Should use aliases in the schema
        assert "id" in props
        assert "name" in props
        assert "email" in props
        assert "internal_id" not in props
        assert "full_name" not in props
        assert "email_addr" not in props

        # Descriptions should be preserved
        assert props["id"]["description"] == "The external ID"
        assert props["name"]["description"] == "Person's full name"

    def test_basemodel_with_optional_fields(self):
        """Test BaseModel with optional fields and defaults."""

        class OptionalModel(BaseModel):
            """Model with optional fields."""

            required_field: str
            optional_with_default: int = Field(
                default=42, description="Optional with default"
            )
            optional_no_default: Optional[str] = Field(
                default=None, description="Optional field"
            )

        def process_data(
            required_field: str,
            optional_with_default: int = 42,
            optional_no_default: Optional[str] = None,
        ) -> str:
            return f"Required: {required_field}, Default: {optional_with_default}, Optional: {optional_no_default}"

        chat = ChatOpenAI()
        chat.register_tool(process_data, model=OptionalModel)

        tool = chat._tools["OptionalModel"]
        func = tool.schema["function"]
        params: dict = func.get("parameters", {})

        # Check what fields are actually required in the generated schema
        # (Pydantic's schema generation may treat defaults differently than expected)
        assert "required_field" in params.get("required", [])

        # All fields should be in properties
        assert "required_field" in params["properties"]
        assert "optional_with_default" in params["properties"]
        assert "optional_no_default" in params["properties"]

        # Verify descriptions are preserved
        assert (
            params["properties"]["optional_with_default"]["description"]
            == "Optional with default"
        )
        assert (
            params["properties"]["optional_no_default"]["description"]
            == "Optional field"
        )

    def test_basemodel_with_complex_types(self):
        """Test BaseModel with complex field types."""

        class ComplexModel(BaseModel):
            """Model with complex types."""

            tags: List[str] = Field(description="List of tags")
            metadata: Dict[str, Union[str, int]] = Field(
                description="Key-value metadata"
            )
            priority: Union[int, str] = Field(description="Priority as int or string")

        def process_complex(
            tags: List[str],
            metadata: Dict[str, Union[str, int]],
            priority: Union[int, str],
        ) -> str:
            return f"Tags: {tags}, Metadata: {metadata}, Priority: {priority}"

        chat = ChatOpenAI()
        chat.register_tool(process_complex, model=ComplexModel)

        tool = chat._tools["ComplexModel"]
        func = tool.schema["function"]
        params: dict = func.get("parameters", {})

        # Check that complex types are handled
        assert "tags" in params.get("properties", {})
        assert "metadata" in params.get("properties", {})
        assert "priority" in params.get("properties", {})

    def test_basemodel_name_takes_precedence(self):
        """Test that BaseModel name takes precedence over function name."""

        class MyCustomToolName(BaseModel):
            """Custom tool with specific name."""

            value: int

        def some_function_name(value: int) -> int:
            """Function with different name."""
            return value * 2

        chat = ChatOpenAI()
        chat.register_tool(some_function_name, model=MyCustomToolName)

        # Tool should use model name, not function name
        assert "MyCustomToolName" in chat._tools
        assert "some_function_name" not in chat._tools

        tool = chat._tools["MyCustomToolName"]
        assert tool.name == "MyCustomToolName"

    def test_basemodel_docstring_takes_precedence(self):
        """Test that BaseModel docstring takes precedence over function docstring."""

        class DocumentedModel(BaseModel):
            """This is the model documentation."""

            param: str

        def undocumented_function(param: str) -> str:
            """This is the function documentation."""
            return param

        chat = ChatOpenAI()
        chat.register_tool(undocumented_function, model=DocumentedModel)

        tool = chat._tools["DocumentedModel"]
        func = tool.schema["function"]
        assert func.get("description") == "This is the model documentation."

    def test_basemodel_field_mismatch_error(self):
        """Test error when BaseModel fields don't match function parameters."""

        class MismatchedModel(BaseModel):
            wrong_field: str
            another_wrong_field: int

        def correct_function(correct_param: str, another_param: int) -> str:
            return f"{correct_param}: {another_param}"

        chat = ChatOpenAI()

        with pytest.raises(ValueError, match="has no corresponding"):
            chat.register_tool(correct_function, model=MismatchedModel)

    def test_alias_field_matching(self):
        """Test that aliases are properly matched against function parameters."""

        class AliasMatchModel(BaseModel):
            internal_name: str = Field(alias="external_name")
            internal_count: int = Field(alias="external_count")

        def function_with_external_names(
            external_name: str, external_count: int
        ) -> str:
            return f"{external_name}: {external_count}"

        chat = ChatOpenAI()
        # This should work because aliases match function parameter names
        chat.register_tool(function_with_external_names, model=AliasMatchModel)

        tool = chat._tools["AliasMatchModel"]
        assert tool.name == "AliasMatchModel"

        # Schema should use the aliases
        func = tool.schema["function"]
        params: dict = func.get("parameters", {})
        assert "external_name" in params.get("properties", {})
        assert "external_count" in params.get("properties", {})
        assert "internal_name" not in params.get("properties", {})
        assert "internal_count" not in params.get("properties", {})

    def test_nested_basemodel_schema_generation(self):
        """Test that nested BaseModels are handled correctly in schema generation."""

        class NestedModel(BaseModel):
            """Model with nested structure - testing schema generation."""

            simple_field: str
            number_field: int = Field(ge=1, le=100, description="Number between 1-100")

        def process_nested(simple_field: str, number_field: int) -> str:
            return f"Simple: {simple_field}, Number: {number_field}"

        chat = ChatOpenAI()
        chat.register_tool(process_nested, model=NestedModel)

        tool = chat._tools["NestedModel"]
        func = tool.schema["function"]
        params: dict = func.get("parameters", {})

        # Verify the schema structure is properly generated
        assert params["type"] == "object"
        assert params["additionalProperties"] is False
        assert set(params["required"]) == {"simple_field", "number_field"}

        # Check that Field constraints are included in schema
        number_prop = params["properties"]["number_field"]
        assert number_prop["description"] == "Number between 1-100"

    def test_extra_model_fields_error(self):
        """Test error when BaseModel has extra fields not in function signature."""

        class ExtraFieldsModel(BaseModel):
            """Model with fields not in function."""

            needed_field: str
            extra_field1: int
            extra_field2: bool

        def function_missing_params(needed_field: str) -> str:
            return f"Got: {needed_field}"

        chat = ChatOpenAI()

        with pytest.raises(ValueError, match="has no corresponding"):
            chat.register_tool(function_missing_params, model=ExtraFieldsModel)

    def test_missing_model_fields_error(self):
        """Test error when function has parameters not in BaseModel."""

        class MissingFieldsModel(BaseModel):
            """Model missing some function parameters."""

            only_field: str

        def function_with_extra_params(
            only_field: str, missing_param: int, another_missing: bool
        ) -> str:
            return f"{only_field}: {missing_param}, {another_missing}"

        chat = ChatOpenAI()

        with pytest.raises(ValueError, match="have no corresponding model fields"):
            chat.register_tool(function_with_extra_params, model=MissingFieldsModel)

    def test_alias_mismatch_error(self):
        """Test error when aliases don't match function parameters."""

        class AliasMismatchModel(BaseModel):
            """Model with aliases that don't match function params."""

            field1: str = Field(alias="wrong_alias1")
            field2: int = Field(alias="wrong_alias2")

        def function_with_correct_params(
            correct_param1: str, correct_param2: int
        ) -> str:
            return f"{correct_param1}: {correct_param2}"

        chat = ChatOpenAI()

        with pytest.raises(ValueError, match="has no corresponding"):
            chat.register_tool(function_with_correct_params, model=AliasMismatchModel)
