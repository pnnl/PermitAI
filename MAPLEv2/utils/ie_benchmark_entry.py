from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union

@dataclass
class IEBenchmarkEntry:
    """
    A dataclass representing an information extraction benchmark entry.
    
    This class is designed for benchmarks that involve extracting specific fields
    from documents, with reference to the expected answer(s).

    Required Attributes:
        field_name (str): The name of the field to extract (e.g., "Categorical Exclusion Applied").
        field_type (str): The output type required for formatting the LLM output
        description (str): Description of the field's meaning and expected format.
        answer (Union[str, List[str]]): The expected answer(s) for the field.

    Optional Attributes:
        entry_id (Optional[str]): A unique identifier for this benchmark entry.
        file_name (Optional[str]): The name of the source file.
        chunks_json (Optional[str]): Path to JSON file containing document chunks.
        document_text (Optional[str]): The text content to extract metadata from.
        page_number (Optional[List[int]]): The page number where the field is located.
        metadata (Dict[str, Any]): Additional metadata for the benchmark entry.
    """
    field_name: str
    field_type: str
    description: str
    answer: Union[str, List[str], float, int]
    
    entry_id: Optional[str] = None
    file_name: Optional[str] = None
    chunks_json: Optional[str] = None
    document_text: Optional[str] = None
    page_number: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Counter for generating entry IDs
    _id_counter = {}
    
    def __post_init__(self):
        """
        Post-initialization processing to ensure data consistency.
        """
        # Ensure file_name format consistency
        if self.file_name and self.file_name.endswith('.pdf') and self.file_name.strip():
            self.file_name = self.file_name[:-4]
            
        # Generate entry_id if not provided
        if not self.entry_id:
            # Get file name without extension
            base_name = self.file_name if self.file_name else 'unknownDoc'
            
            # Use the field name as part of the ID
            field_part = self.field_name.lower().replace(' ', '_')
            
            # Create counter key for this file+field combination
            counter_key = f"{base_name}_{field_part}"
            
            # Increment counter for this file+field combination
            if counter_key not in self.__class__._id_counter:
                self.__class__._id_counter[counter_key] = 0
            self.__class__._id_counter[counter_key] += 1
            
            # Create ID with format: file_name_field_counter
            self.entry_id = f"{counter_key}_{self.__class__._id_counter[counter_key]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entry to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the entry.
        """
        return {
            'field_name': self.field_name,
            'field_type': self.field_type,
            'description': self.description,
            'answer': self.answer,
            'entry_id': self.entry_id,
            'file_name': self.file_name,
            'chunks_json': self.chunks_json,
            'document_text': self.document_text,
            'page_number': self.page_number,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IEBenchmarkEntry':
        """
        Create an entry from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing entry data.
            
        Returns:
            IEBenchmarkEntry: New instance created from the dictionary.
        """
        # Extract core fields
        core_fields = {k: v for k, v in data.items() 
                      if k in ['field_name', 'field_type', 'description', 'answer', 'entry_id', 
                              'file_name', 'chunks_json', 'document_text', 'page_number']}
        
        # Extract metadata (any fields not in the core fields)
        metadata = {k: v for k, v in data.items() 
                   if k not in core_fields and k != 'metadata'}
        
        # Add explicit metadata if provided
        if 'metadata' in data and isinstance(data['metadata'], dict):
            metadata.update(data['metadata'])
            
        return cls(**core_fields, metadata=metadata)
    
    def update_field(self, attr_name: str, value: Any) -> None:
        """
        Update a attribute value.
        
        Args:
            attr_name (str): Name of the attribute to update.
            value (Any): New value for the attribute.
            
        Raises:
            AttributeError: If the attribute doesn't exist.
        """
        if hasattr(self, attr_name):
            setattr(self, attr_name, value)
        elif attr_name in self.metadata:
            self.metadata[attr_name] = value
        else:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{attr_name}'")