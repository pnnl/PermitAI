from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class CommentClassificationBenchmarkEntry:
    """
    A benchmark entry for the Comment-Classification task.

    Required Attributes:
        in_scope (bool): boolean flag to denote in-scope or out of scope
        comment (str): the full public-comment text

    Optional Attributes:
        project_description: Optional[str] - the description of the project
        project_file : Optional[str]  — the file path to the project description text
        entry_id  : Optional[str]  — A unique identifier for this benchmark entry.
        metadata  : Dict[str, Any] — Additional metadata for the benchmark entry.
    """
    # ----- required -----
    comment: str
    in_scope: bool

    # ----- optional -----
    project_description: Optional[str] = None
    project_file: Optional[str] = None
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
            self.entry_id = f"ComClass_{CommentClassificationBenchmarkEntry._id_counter}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "in_scope": self.in_scope,
            "comment": self.comment,
            "project_description": self.project_description,
            "project_file": self.project_file,
            "entry_id": self.entry_id,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommentClassificationBenchmarkEntry":
        """
        Create an entry from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing entry data.
            
        Returns:
            CommentClassificationBenchmarkEntry: New instance created from the dictionary.
        """
        # Extract core fields
        core_fields = {k: v for k, v in data.items() 
                      if k in ['in_scope', 'comment', 'project_description', 
                               'project_file', 'entry_id'
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