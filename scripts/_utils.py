import collections.abc
import inspect
import re
import subprocess
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    _GenericAlias,  # type: ignore
    get_type_hints,
)

TYPING_IMPORTS = {
    "List",
    "Dict",
    "Tuple",
    "Set",
    "FrozenSet",
    "Deque",
    "DefaultDict",
    "OrderedDict",
    "Counter",
    "ChainMap",
    "Iterable",
    "Iterator",
    "Generator",
    "Sequence",
    "MutableSequence",
    "Mapping",
    "MutableMapping",
    "AsyncIterable",
    "AsyncIterator",
    "AsyncGenerator",
    "Reversible",
    "Container",
    "Collection",
    "Callable",
    "AbstractSet",
    "MutableSet",
    "Hashable",
    "Sized",
    "Union",
    "Optional",
    "Literal",
    "Final",
    "ClassVar",
    "TypeVar",
    "Generic",
    "Protocol",
    "NamedTuple",
    "TypedDict",
    "NewType",
    "Any",
    "AnyStr",
    "NoReturn",
    "Type",
    "Coroutine",
    "Awaitable",
    "AsyncContextManager",
    "ContextManager",
}


def create_typeddict_for_method(
    method: Callable,
    class_name: str,
    excluded_fields: Optional[set[str]] = None,
    localns: Optional[dict[str, Any]] = None,
) -> tuple[str, set[str]]:
    type_hints = get_type_hints(method, localns=localns)
    annotations = {}
    for param_name, param in inspect.signature(method).parameters.items():
        if param.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            if param_name not in (excluded_fields or {}):
                annotations[param_name] = type_hints.get(param_name, Any)

    code = f"class {class_name}(TypedDict, total=False):\n"
    imports = set()
    for param_name, param_type in annotations.items():
        type_str, new_imports = get_type_string(param_type)
        imports.update(new_imports)
        code += f"    {param_name}: {type_str}\n"

    return code, imports


def get_type_string(typ: Any) -> tuple[str, set[str]]:
    imports = set()
    if typ is type(None):
        return "None", imports
    if isinstance(typ, _GenericAlias) or hasattr(typ, "__origin__"):
        origin = typ.__origin__
        if origin is Union:
            args = []
            for arg in typ.__args__:
                arg_str, arg_imports = get_type_string(arg)
                args.append(arg_str)
                imports.update(arg_imports)
            if len(args) == 2 and "None" in args:
                non_none = next(arg for arg in args if arg != "None")
                imports.add("from typing import Optional")
                return f"Optional[{non_none}]", imports
            imports.add("from typing import Union")
            return f"Union[{', '.join(args)}]", imports
        if origin is Literal:
            args = ", ".join(repr(arg) for arg in typ.__args__)
            imports.add("from typing import Literal")
            return f"Literal[{args}]", imports
        if origin is collections.abc.Callable:
            type_args = typ.__args__ or ()

            # typing.Callable is typically represented as (Ellipsis, R) or (A, B, R)
            # (older versions may use ([A, B], R)).
            if not type_args:
                imports.add("from typing import Callable")
                return "Callable[..., Any]", imports

            ret_type = type_args[-1]
            params = type_args[:-1]

            # Handle Callable[..., R]
            if len(type_args) == 2 and type_args[0] is Ellipsis:
                ret_str, ret_imports = get_type_string(ret_type)
                imports.update(ret_imports)
                imports.add("from typing import Callable")
                return f"Callable[..., {ret_str}]", imports

            # Handle Callable[[A, B], R]
            if len(type_args) == 2 and isinstance(type_args[0], (list, tuple)):
                params = tuple(type_args[0])

            param_strs = []
            for arg in params:
                arg_str, arg_imports = get_type_string(arg)
                param_strs.append(arg_str)
                imports.update(arg_imports)

            ret_str, ret_imports = get_type_string(ret_type)
            imports.update(ret_imports)
            imports.add("from typing import Callable")
            return f"Callable[[{', '.join(param_strs)}], {ret_str}]", imports
        origin_name = origin.__name__
        args = []
        for arg in typ.__args__:
            arg_str, arg_imports = get_type_string(arg)
            args.append(arg_str)
            imports.update(arg_imports)
        if origin_name in TYPING_IMPORTS:
            imports.add(f"from typing import {origin_name}")
        elif getattr(origin, "__module__", "builtins") not in (
            "builtins",
            "typing",
            "typing_extensions",
            "collections.abc",
        ):
            module = origin.__module__
            imports.add(f"import {module}")
            origin_name = f"{module}.{origin_name}"
        return f"{origin_name}[{', '.join(args)}]", imports
    elif isinstance(typ, type):
        module = typ.__module__
        if module == "builtins":
            return typ.__name__, imports
        elif module == "typing":
            if typ.__name__ in TYPING_IMPORTS:
                imports.add(f"from typing import {typ.__name__}")
            return typ.__name__, imports
        else:
            imports.add(f"import {module}")
            return f"{module}.{typ.__name__}", imports
    else:
        type_str = str(typ)
        if type_str == "Ellipsis":
            return "...", imports
        typing_match = re.match(r"typing\.(\w+)(\[.*\])?", type_str)
        if typing_match:
            type_name = typing_match.group(1)
            if type_name in TYPING_IMPORTS:
                imports.add(f"from typing import {type_name}")
            return type_str.replace("typing.", ""), imports

        # Handle namespaced types (including union syntax)
        if "|" in type_str:
            parts = [part.strip() for part in type_str.split("|")]
            processed_parts = []
            for part in parts:
                namespace_match = re.match(
                    r"([a-zA-Z_][a-zA-Z0-9_.]*)\.([a-zA-Z_][a-zA-Z0-9_]*)",
                    part,
                )
                if namespace_match:
                    namespace, type_name = namespace_match.groups()
                    imports.add(f"import {namespace}")
                    processed_parts.append(f"{namespace}.{type_name}")
                else:
                    processed_parts.append(part)
            return " | ".join(processed_parts), imports

        # Handle namespaced types
        namespace_match = re.match(
            r"([a-zA-Z_][a-zA-Z0-9_.]*)\.([a-zA-Z_][a-zA-Z0-9_]*)",
            type_str,
        )
        if namespace_match:
            namespace, type_name = namespace_match.groups()
            imports.add(f"import {namespace}")
            return f"{namespace}.{type_name}", imports

        return type_str, imports


def generate_typeddict_code(
    method: Callable,
    class_name: str,
    excluded_fields: Optional[set[str]] = None,
    localns: Optional[dict[str, Any]] = None,
) -> str:
    typeddict, imports = create_typeddict_for_method(
        method, class_name, excluded_fields, localns
    )
    imports_code = "from typing import TypedDict, Any\n"
    imports_code += "\n".join(sorted(imports)) + "\n\n"
    return imports_code + typeddict


def write_code_to_file(code: str, path: Path, setup_code: Optional[str] = None):
    code_ = f"""
# ---------------------------------------------------------
# Do not modify this file. It was generated by `scripts/generate_typed_dicts.py`.
# ---------------------------------------------------------

{setup_code or ""}

{code}
"""

    with open(path, "w") as f:
        f.write(code_)

    subprocess.run(
        ["ruff", "format", str(path), "--config", "pyproject.toml"],
        check=False,
    )
    subprocess.run(
        ["ruff", "check", "--fix", str(path), "--config", "pyproject.toml"],
        check=False,
    )
