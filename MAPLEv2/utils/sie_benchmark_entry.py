from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List


@dataclass
class StructuredIEBenchmarkEntry:
    """
    A dataclass representing a benchmark entry for structured information extraction task from document.
    
    This class is designed for benchmarks that involve analyzing documents 
    and extracting structured information from them.

    Required Attributes:
        text (str): The text content.
        info (Dict[str, Any]): Dictionary containing the expected info/metadata.
        schema_json (str): Path to the JSON file with the output schema
        
    Optional Attributes:
        instruction_block (Optional[str]): Instruction for each field in the schema
        entry_id (Optional[str]): A unique identifier for this benchmark entry.
    """
    text: str
    info: Union[List[Dict[str, Any]], Dict[str, Any]]
    schema_json: str
    instruction_block: Optional[str] = None
    entry_id: Optional[str] = None
    
    # Counter for generating entry IDs
    _id_counter = 0
    
    def __post_init__(self):
        """
        Post-initialization processing to ensure data consistency.
        """
        # Generate entry_id if not provided
        if not self.entry_id:
            # Use class variable to track counter
            self.__class__._id_counter += 1
            self.entry_id = f"sie_{self.__class__._id_counter}"
        
        # Ensure info is a dictionary
        if not isinstance(self.info, dict) and not isinstance(self.info, list):
            raise TypeError(f"info must be a dictionary or list, got {type(self.info)}")
        
        # Ensure notice_text is not empty
        if not self.text or not self.text.strip():
            raise ValueError("text cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entry to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the entry.
        """
        return {
            'text': self.text,
            'info': self.info,
            'schema_json': self.schema_json,
            'instruction_block': self.instruction_block,
            'entry_id': self.entry_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructuredIEBenchmarkEntry':
        """
        Create an entry from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing entry data.
            
        Returns:
            StructuredIEBenchmarkEntry: New instance created from the dictionary.
        """
        # Extract core fields
        core_fields = {k: v for k, v in data.items() 
                      if k in ['text', 'info', 'schema_json', 'instruction_block', 'entry_id']}
            
        return cls(**core_fields)
    
    def update_field(self, field_name: str, value: Any) -> None:
        """
        Update a field value.
        
        Args:
            field_name (str): Name of the field to update.
            value (Any): New value for the field.
            
        Raises:
            AttributeError: If the field doesn't exist.
        """
        if hasattr(self, field_name):
            setattr(self, field_name, value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{field_name}'")