from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class TribalBenchmarkEntry:
    """
    A dataclass representing a tribal benchmark entry for document analysis.
    
    This class is designed for benchmarks that involve identifying and extracting
    tribal information from documents, particularly useful for environmental
    impact assessments and NEPA compliance documents.

    Required Attributes:
        file_name (str): The name of the source file.
        source_type (str): The type of source (either 'name' or 'number').
        source_caption (str): The caption or description of the source.
        tribes (List[str]): List of tribal names identified in the source.
        
    Optional Attributes:
        document_text (str): The text content of the document or section.
    """
    file_name: str
    source_type: str
    source_caption: str
    tribes: List[str]
    entry_id: Optional[str] = None
    document_text: Optional[str] = None

    # Counter for generating entry IDs
    _id_counter = 0
    
    def __post_init__(self):
        """
        Post-initialization processing to ensure data consistency.
        """
        # Ensure file_name format consistency
        if self.file_name and not self.file_name.endswith('.pdf') and self.file_name.strip():
            self.file_name = f"{self.file_name}.pdf"
        
        # Generate entry_id if not provided
        if not self.entry_id:
            # Create counter key for this file+field combination
            counter_key = "tribe_data"
            
            # Increment counter for this file+field combination
            self.__class__._id_counter += 1
            
            # Create ID with format: file_name_field_counter
            self.entry_id = f"{counter_key}_{self.__class__._id_counter}"
        
        # Validate source_type
        if self.source_type not in ['name', 'number']:
            raise ValueError(f"source_type must be either 'name' or 'number', got '{self.source_type}'")
        
        # Ensure tribes is a list
        if isinstance(self.tribes, str):
            # If tribes is a single string, convert to list
            self.tribes = [self.tribes] if self.tribes.strip() else []
        elif self.tribes is None:
            self.tribes = []
        
        # Clean and validate tribe names
        self.tribes = [tribe.strip() for tribe in self.tribes if tribe and tribe.strip()]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entry to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the entry.
        """
        return {
            'file_name': self.file_name,
            'source_type': self.source_type,
            'source_caption': self.source_caption,
            'tribes': self.tribes,
            'entry_id': self.entry_id,
            'document_text': self.document_text
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TribalBenchmarkEntry':
        """
        Create an entry from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing entry data.
            
        Returns:
            TribalBenchmarkEntry: New instance created from the dictionary.
        """
        # Extract core fields
        core_fields = {k: v for k, v in data.items() 
                      if k in ['file_name', 'source_type', 'source_caption', 'tribes', 'entry_id', 'document_text']}
            
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