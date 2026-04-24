from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class QABenchmarkEntry:
    """
    A dataclass representing a single entry in the question-answering benchmark dataset.

    Required Attributes:
        question (str): The actual question text.
        answer (str): The correct answer to the question.

    Optional Attributes:
        file_name (Optional[str]): The name of the file containing the entry.
        page_number (Optional[int]): The page number where the entry is located.
        question_type (Optional[str]): The type of question (e.g., 'closed', 'open', 'evaluation').
        proof (Optional[str]): The proof or explanation for the answer.
        context (Optional[str]): The context or background information for the question.
        entry_id (Optional[str]): A unique identifier for the benchmark entry.
        chunks_json (Optional[str]): Path to the JSON file with text chunks, required for PDF context.
        metadata (Optional[Dict[str, Any]]): Additional metadata for the question.
    """
    question: str
    answer: str
    file_name: Optional[str] = None
    page_number: Optional[int] = None
    question_type: Optional[str] = None
    proof: Optional[str] = None
    context: Optional[str] = None
    entry_id: Optional[str] = None
    chunks_json: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Counter for generating question IDs
    _id_counter = {}
    
    def __post_init__(self):
        """
        Post-initialization processing to ensure data consistency.
        """
        # Ensure file_name format consistency
        if self.file_name and self.file_name.endswith('.pdf') and self.file_name.strip():
            self.file_name = self.file_name[:-4]
            
        # Generate question_id if not provided
        if not self.entry_id:
            # Get file name without extension
            file_prefix = self.file_name if self.file_name else 'unknownDoc'
            
            # Get question type or use 'general' if not specified
            question_type = self.question_type if self.question_type else 'unknownType'
            
            # Create key for counter
            counter_key = f"{file_prefix}_{question_type}"
            
            # Increment counter for this file+type combination
            if counter_key not in self.__class__._id_counter:
                self.__class__._id_counter[counter_key] = 0
            self.__class__._id_counter[counter_key] += 1
            
            # Create ID with format: file_name_question_type_counter
            self.entry_id = f"{counter_key}_{self.__class__._id_counter[counter_key]}"
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entry to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the entry.
        """
        return {
            'question': self.question,
            'answer': self.answer,
            'file_name': self.file_name,
            'chunks_json': self.chunks_json,
            'page_number': self.page_number,
            'question_type': self.question_type,
            'proof': self.proof,
            'context': self.context,
            'question_id': self.entry_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QABenchmarkEntry':
        """
        Create an entry from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing entry data.
            
        Returns:
            QABenchmarkEntry: New instance created from the dictionary.
        """
        # Extract core fields
        core_fields = {k: v for k, v in data.items() 
                      if k in ['question', 'answer', 'file_name', 'page_number', 'chunks_json',
                              'question_type', 'proof', 'context', 'question_id']}
        
        # Extract metadata (any fields not in the core fields)
        metadata = {k: v for k, v in data.items() 
                   if k not in core_fields and k != 'metadata'}
        
        # Add explicit metadata if provided
        if 'metadata' in data and isinstance(data['metadata'], dict):
            metadata.update(data['metadata'])
            
        return cls(**core_fields, metadata=metadata)
    
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
        elif field_name in self.metadata:
            self.metadata[field_name] = value
        else:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{field_name}'")