#!/usr/bin/env python3
"""
JSON Schema to Pydantic Class Generator

This script converts JSON schema definitions to Pydantic model classes.
It's designed to work with the Maple evaluation framework's data structures.

Usage:
    python schema_utils.py --input schema.json --output models.py --class-name ProjectData
    
    # Or use as a module:
    from utils.schema_utils import JSONToPydanticGenerator
    generator = JSONToPydanticGenerator()
    pydantic_code = generator.generate_from_dict(schema_dict, "ProjectData")
"""

import json
import sys
import os
from typing import Dict, Type, Any, Set, List, Union, Literal, get_args, get_origin
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from enum import Enum
import inspect

# Add parent directory to path for imports when running as standalone
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_schema_utils.log")

logger = LogManager.get_logger("schema_utils")


class HierarchicalPydanticGenerator:
    """
    Converts hierarchical JSON schema definitions to complex Pydantic model classes.
    
    This generator creates Pydantic BaseModel classes with:
    - Nested model references
    - Model inheritance
    - Literal types for enums
    - List and Optional types
    - Complex field relationships
    """
    
    def __init__(self):
        """Initialize the generator."""
        self.imports = {
            "pydantic": ["BaseModel", "Field"],
            "typing": ["Optional", "List", "Dict", "Any", "Union", "Literal"]
        }
        self.models = {}  # Store all model definitions
        self.model_dependencies = {}  # Track dependencies between models
        self.executed_classes = {} # Executed classes for dynamic pydantic class generation
        logger.info("HierarchicalPydanticGenerator initialized")
    
    def _determine_python_type(self, field_config: Dict[str, Any]) -> str:
        """
        Determine Python type annotation from field configuration.
        
        Args:
            field_config: Configuration dictionary for the field
            
        Returns:
            str: Python type annotation
        """
        field_type = field_config.get("type", "str")
        is_optional = field_config.get("optional", True)
        is_list = field_config.get("list", False)
        union_types = field_config.get("union_types", [])
        
        # Handle union types (e.g., Union[str, List[str]])
        if union_types:
            union_str = ", ".join(union_types)
            base_type = f"Union[{union_str}]"
        # Handle literal types
        elif field_type == "literal" and "values" in field_config:
            values_str = ", ".join([f'"{v}"' for v in field_config["values"]])
            base_type = f"Literal[{values_str}]"
        # Handle model references
        elif field_type == "model":
            model_name = field_config.get("model_name", "BaseModel")
            base_type = model_name
        # Handle basic types
        elif field_type == "str":
            base_type = "str"
        elif field_type == "int":
            base_type = "int"
        elif field_type == "float":
            base_type = "float"
        elif field_type == "bool":
            base_type = "bool"
        else:
            base_type = "str"
        
        # Wrap in List if needed
        if is_list:
            base_type = f"List[{base_type}]"
        
        # Wrap in Optional if needed
        if is_optional and not is_list:
            base_type = f"Optional[{base_type}]"
        
        return base_type
    
    def _format_field_args(self, field_config: Dict[str, Any]) -> str:
        """
        Format Field() arguments from field configuration.
        
        Args:
            field_config: Configuration dictionary for the field
            
        Returns:
            str: Formatted Field() arguments
        """
        args = []
        
        # Examples (keep for validation)
        if "examples" in field_config:
            examples = field_config["examples"]
            if isinstance(examples, list):
                examples_str = "[" + ", ".join([f'"{ex}"' for ex in examples]) + "]"
            else:
                examples_str = f'["{examples}"]'
            args.append(f"examples={examples_str}")
        
        # Default value
        if "default" in field_config:
            default = field_config["default"]
            if isinstance(default, str):
                if default == "":
                    args.append(f"default=''")
                else:
                    args.append(f'default="{default}"')
            elif isinstance(default, list):
                args.append(f"default={default}")
            # elif isinstance(default, dict):
            #     args.append(f"default={{}}")
            # elif default is None:
            #     args.append(f"default=None")
            else:
                args.append("")
        
        return ", ".join(args)
    
    def _sanitize_class_name(self, name: str) -> str:
        """Sanitize class name to be a valid Python identifier."""
        # Convert to PascalCase and remove invalid characters
        sanitized = "".join(word.capitalize() for word in name.replace("_", " ").split())
        sanitized = "".join(c if c.isalnum() else "" for c in sanitized)
        
        if not sanitized or not sanitized[0].isalpha():
            sanitized = f"Model{sanitized}"
        
        return sanitized
    
    def _sanitize_field_name(self, field_name: str) -> str:
        """Sanitize field name to be a valid Python identifier."""
        # Keep original case for field names, just ensure validity
        sanitized = "".join(c if c.isalnum() else "_" for c in field_name)
        
        if not sanitized[0].isalpha() and sanitized[0] != "_":
            sanitized = f"field_{sanitized}"
        
        # Handle Python keywords
        python_keywords = {
            "class", "def", "if", "else", "for", "while", "try", "except",
            "import", "from", "as", "with", "return", "yield", "lambda",
            "and", "or", "not", "in", "is", "None", "True", "False"
        }
        
        if sanitized.lower() in python_keywords:
            sanitized = f"{sanitized}_field"
        
        return sanitized
    
    def _extract_model_dependencies(self, model_config: Dict[str, Any]) -> Set[str]:
        """Extract model dependencies from a model configuration."""
        dependencies = set()
        
        for field_name, field_config in model_config.get("fields", {}).items():
            if field_config.get("type") == "model":
                model_name = field_config.get("model_name")
                if model_name:
                    dependencies.add(model_name)
        
        # Add inheritance dependencies
        if "inherits" in model_config:
            dependencies.add(model_config["inherits"])
        
        return dependencies
    
    def _topological_sort(self) -> List[str]:
        """Sort models in dependency order (dependencies first)."""
        # Build dependency graph
        in_degree = {model: 0 for model in self.models}
        graph = {model: [] for model in self.models}
        
        for model, deps in self.model_dependencies.items():
            for dep in deps:
                if dep in self.models:  # Only consider dependencies that are in our model set
                    graph[dep].append(model)
                    in_degree[model] += 1
        
        # Topological sort using Kahn's algorithm
        queue = [model for model, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(self.models):
            # Circular dependency detected, return original order
            logger.warning("Circular dependency detected in models")
            return list(self.models.keys())
        
        return result
    
    def _generate_single_model(self, model_name: str, model_config: Dict[str, Any]) -> str:
        """Generate code for a single Pydantic model."""
        lines = []
        
        # Class definition with inheritance
        inherits = model_config.get("inherits")
        if inherits:
            class_def = f"class {model_name}({inherits}):"
        else:
            class_def = f"class {model_name}(BaseModel):"
        
        lines.append(class_def)
        
        # Fields without descriptions
        fields = model_config.get("fields", {})
        if not fields and not inherits:
            lines.append("    pass")
        else:
            for field_name, field_config in fields.items():
                sanitized_name = self._sanitize_field_name(field_name)
                field_type = self._determine_python_type(field_config)
                field_args = self._format_field_args(field_config)
                
                if field_args:
                    field_line = f"    {sanitized_name}: {field_type} = Field({field_args})"
                else:
                    field_line = f"    {sanitized_name}: {field_type}"
                
                lines.append(field_line)
        
        return "\n".join(lines)
    
    def generate_from_dict(self, schema_dict: Dict[str, Any]) -> str:
        """
        Generate complete Pydantic code from hierarchical schema.
        
        Args:
            schema_dict: Hierarchical schema dictionary
            
        Returns:
            str: Complete Pydantic module code
        """
        logger.info(f"Generating hierarchical Pydantic models from schema with {len(schema_dict.get('models', {}))} models")
        
        # Store models and extract dependencies
        self.models = schema_dict.get("models", {})
        self.model_dependencies = {}
        
        for model_name, model_config in self.models.items():
            self.model_dependencies[model_name] = self._extract_model_dependencies(model_config)
        
        # Sort models in dependency order
        sorted_models = self._topological_sort()
        
        # Generate code
        lines = []
        
        # Add imports
        lines.append("from pydantic import BaseModel, Field")
        lines.append("from typing import Optional, List, Dict, Any, Union, Literal")
        lines.append("")
        
        # Generate each model
        for model_name in sorted_models:
            model_config = self.models[model_name]
            model_code = self._generate_single_model(model_name, model_config)
            lines.append(model_code)
            lines.append("")  # Empty line between models
        
        generated_code = "\n".join(lines)
        logger.info(f"Successfully generated hierarchical Pydantic code with {len(self.models)} models")
        
        return generated_code
    
    def generate_from_json_file(self, json_file: str) -> str:
        """Generate Pydantic code from JSON file."""
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                schema_dict = json.load(f)
            
            return self.generate_from_dict(schema_dict)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON file: {json_file} - {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error reading JSON file: {str(e)}")
            raise
    
    def save_to_file(self, code: str, output_file: str) -> None:
        """Save generated code to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        logger.info(f"Hierarchical Pydantic models saved to: {output_file}")

    def exec_with_globals(self, pydantic_code: str, class_name: str) -> Type[BaseModel]:
        """
        Args:
            pydantic_code (str): Generated Pydantic class code
            class_name (str): Name of the class to extract
            
        Returns:
            Type[BaseModel]: The dynamically created Pydantic class
        """
        # Create a controlled namespace with required imports
        namespace = {
            '__builtins__': __builtins__,
            'BaseModel': BaseModel,
            'Field': None,  # Will be imported by the code
            'Optional': None,  # Will be imported by the code
            'List': None,
            'Dict': None,
            'Any': None,
            'Union': None,
        }
        
        try:
            # Execute the code in the controlled namespace
            exec(pydantic_code, namespace)
            
            # Extract the class from the namespace
            if class_name not in namespace:
                raise ValueError(f"Class '{class_name}' not found in generated code")
            
            pydantic_class = namespace[class_name]
            
            # Verify it's a Pydantic model
            if not issubclass(pydantic_class, BaseModel):
                raise ValueError(f"Generated class is not a Pydantic BaseModel")
            
            # Cache for later use
            self.executed_classes[class_name] = pydantic_class
            
            return pydantic_class
            
        except Exception as e:
            raise RuntimeError(f"Error executing Pydantic code: {str(e)}")
        
def generate_instruction(json_file: Dict[str, Any]) -> str:
    """Generate a compact documentation format optimized for LLM prompts."""
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            schema_dict = json.load(f)
        
        models = schema_dict.get("models", {})
        doc_lines = []
        
        doc_lines.append("Extract the following information and structure it as JSON:")
        doc_lines.append("")
        
        # Focus on the main Output model structure
        output_model = models.get("Output", {})
        if output_model:
            doc_lines.append("Main Structure:")
            fields = output_model.get("fields", {})
            for field_name, field_config in fields.items():
                desc = field_config.get("description", "")
                doc_lines.append(f"- {field_name}: {desc}")
            doc_lines.append("")
        
        # Provide details for complex nested structures
        for model_name, model_config in models.items():
            if model_name == "Output":
                continue
                
            doc_lines.append(f"{model_name} fields:")
            fields = model_config.get("fields", {})
            for field_name, field_config in fields.items():
                desc = field_config.get("description", field_name)
                
                # Add examples if available
                examples_text = ""
                if "examples" in field_config:
                    examples = field_config["examples"]
                    if isinstance(examples, list) and examples:
                        examples_text = f" (e.g., {examples[0]})"
                    elif examples:
                        examples_text = f" (e.g., {examples})"
                
                # Add allowed values for literals
                values_text = ""
                if field_config.get("type") == "literal" and "values" in field_config:
                    values = [v for v in field_config["values"] if v]  # Filter empty strings
                    if values:
                        values_text = f" [Options: {', '.join(values)}]"
                
                doc_lines.append(f"  - {field_name}: {desc}{examples_text}{values_text}")
            
            doc_lines.append("")
        
        return "\n".join(doc_lines)
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON file: {json_file} - {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error reading JSON file: {str(e)}")
        raise

def replace_nulls_with_defaults(
    data: dict[str, Any],
    schema: Type[BaseModel]
) -> dict[str, Any]:
    """
    Recursively replace null values in data with default values from a Pydantic schema.
    Handles nested Pydantic models.
    
    Args:
        data: Dictionary loaded from JSON file
        schema: Pydantic model class with default values
        
    Returns:
        Dictionary with nulls replaced by schema defaults
    """
    result = data.copy()
    
    # Get the schema fields with their defaults
    for field_name, field_info in schema.model_fields.items():
        if field_name not in result:
            continue
            
        field_value = result[field_name]
        field_type = field_info.annotation

        # Unwrap Optional/Union types to get actual type
        actual_type = field_type
        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                actual_type = non_none_args[0]
        
        # Handle null values
        if field_value is None:
            # Check if it's a BaseModel type - instantiate it with defaults
            if isinstance(actual_type, type) and issubclass(actual_type, BaseModel):
                # Create empty instance and convert to dict
                result[field_name] = actual_type().model_dump()
            # Check if it's a List type
            elif get_origin(actual_type) is list:
                result[field_name] = []
            # Use the field's default
            elif field_info.default is not None:
                result[field_name] = field_info.default
            else:
                result[field_name] = None
        
        # Handle nested dicts (recurse)
        elif isinstance(field_value, dict):
            if isinstance(actual_type, type) and issubclass(actual_type, BaseModel):
                result[field_name] = replace_nulls_with_defaults(field_value, actual_type)
        
        # Handle lists
        elif isinstance(field_value, list):
            origin = get_origin(actual_type)
            if origin is list:
                args = get_args(actual_type)
                if args:
                    item_type = args[0]
                    item_origin = get_origin(item_type)
                    if item_origin is Union:
                        item_args = get_args(item_type)
                        non_none_item_args = [arg for arg in item_args if arg is not type(None)]
                        if len(non_none_item_args) == 1:
                            item_type = non_none_item_args[0]
                    
                    if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                        result[field_name] = [
                            replace_nulls_with_defaults(item, item_type) if isinstance(item, dict) else item
                            for item in field_value
                        ]
    
    return result

def pydantic_to_vertex_schema(model_class: type[BaseModel]) -> Dict[str, Any]:
    """
    Convert a Pydantic BaseModel to Vertex AI compatible JSON schema
    """
    def get_field_schema(field_type: Any, field_info: FieldInfo = None) -> Dict[str, Any]:
        """Convert a field type to JSON schema format"""
        
        # Handle Optional types (Union[T, None])
        origin = get_origin(field_type)
        args = get_args(field_type)

        # Handle Literal types
        if origin is Literal:
            # Get the literal values
            literal_values = list(args)
            
            # Determine the type based on the first value
            if literal_values:
                first_value = literal_values[0]
                if isinstance(first_value, str):
                    json_type = "string"
                elif isinstance(first_value, int):
                    json_type = "integer"
                elif isinstance(first_value, float):
                    json_type = "number"
                elif isinstance(first_value, bool):
                    json_type = "boolean"
                else:
                    json_type = "string"  # fallback
            else:
                json_type = "string"
            
            schema = {
                "type": json_type,
                "enum": literal_values
            }
            
            if field_info and hasattr(field_info, 'description') and field_info.description:
                schema["description"] = field_info.description
            
            return schema
        
        if origin is Union:
            # Check if it's Optional (Union[T, None])
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                return get_field_schema(non_none_type, field_info)
            else:
                # Handle other Union types - use the first type
                return get_field_schema(args[0], field_info)
        
        # Handle List types
        if origin is list or field_type is list:
            item_type = args[0] if args else str
            schema = {
                "type": "array",
                "items": get_field_schema(item_type)
            }
            return schema
        
        # Handle Dict types
        if origin is dict or field_type is dict:
            value_type = args[1] if len(args) > 1 else str
            return {
                "type": "object",
                "additionalProperties": get_field_schema(value_type)
            }
        
        # Handle Enum types
        if inspect.isclass(field_type) and issubclass(field_type, Enum):
            return {
                "type": "string",
                "enum": [e.value for e in field_type]
            }
        
        # Handle nested Pydantic models
        if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            return convert_model_to_schema(field_type)
        
        # Handle basic types
        type_mapping = {
            str: "string",
            int: "integer", 
            float: "number",
            bool: "boolean",
            bytes: "string"  # Base64 encoded
        }
        
        json_type = type_mapping.get(field_type, "string")
        schema = {"type": json_type}
        
        # Add field constraints from Field()
        if field_info:
            if hasattr(field_info, 'description') and field_info.description:
                schema["description"] = field_info.description
            
            # Handle numeric constraints
            if json_type in ["integer", "number"]:
                if hasattr(field_info, 'ge') and field_info.ge is not None:
                    schema["minimum"] = field_info.ge
                if hasattr(field_info, 'le') and field_info.le is not None:
                    schema["maximum"] = field_info.le
                if hasattr(field_info, 'gt') and field_info.gt is not None:
                    schema["exclusiveMinimum"] = field_info.gt
                if hasattr(field_info, 'lt') and field_info.lt is not None:
                    schema["exclusiveMaximum"] = field_info.lt
            
            # Handle string constraints
            if json_type == "string":
                if hasattr(field_info, 'min_length') and field_info.min_length is not None:
                    schema["minLength"] = field_info.min_length
                if hasattr(field_info, 'max_length') and field_info.max_length is not None:
                    schema["maxLength"] = field_info.max_length
                if hasattr(field_info, 'pattern') and field_info.pattern is not None:
                    schema["pattern"] = field_info.pattern
        
        return schema
    
    def convert_model_to_schema(model: type[BaseModel]) -> Dict[str, Any]:
        """Convert a Pydantic model to JSON schema object"""
        properties = {}
        required = []
        
        for field_name, field in model.model_fields.items():
            field_schema = get_field_schema(field.annotation, field)
            properties[field_name] = field_schema
            
            # Check if field is required
            if field.is_required():
                required.append(field_name)
        
        schema = {
            "type": "object",
            "properties": properties
        }
        
        if required:
            schema["required"] = required
            
        return schema
    
    # Convert the main model
    return convert_model_to_schema(model_class)
    