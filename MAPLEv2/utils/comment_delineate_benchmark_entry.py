from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class CommentDelineateBenchmarkEntry:
    """
    A benchmark entry for the Comment-Delineation task.

    Required Attributes:
        comment_quote (List[str]): the list of dilineated public comment quotes
        comment_file (str): the path to the file containing the entire comment

    Optional Attributes:
        full_comment : Optional[str]  — the entire comment as a text string
        entry_id  : Optional[str]  — A unique identifier for this benchmark entry.
        metadata  : Dict[str, Any] — Additional metadata for the benchmark entry.
    """

    # ----- required -----
    comment_quotes: List[str]
    comment_file: str

    # ----- optional -----
    full_comment: Optional[str] = None
    entry_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Counter for generating entry IDs
    _id_counter: int = 0

    def __post_init__(self):
        """
        Generate a simple numeric entry_id if none was provided.
        """
        if not self.entry_id:
            self.__class__._id_counter += 1
            self.entry_id = f"Delineate_{CommentDelineateBenchmarkEntry._id_counter}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comment_quotes": self.comment_quotes,
            "comment_file": self.comment_file,
            "full_comment": self.full_comment,
            "entry_id": self.entry_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommentDelineateBenchmarkEntry':
        """
        Create an entry from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing entry data.
            
        Returns:
            CommentDelineateBenchmarkEntry: New instance created from the dictionary.
        """
        # Extract core fields
        core_fields = {k: v for k, v in data.items() 
                      if k in ['comment_quotes', 'comment_file', 
                               'full_comment', 'entry_id'
                               ]}
        
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