from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class CommentBinBenchmarkEntry:
    """
    A benchmark entry for the Comment-Bin-Assignment task.

    Required Attributes:
        bin (str): the correct bin label for this comment
        comment (str): the full public-comment text
        binning_guidance_file (str): json file with binning guidance

    Optional Attributes:
        concern : Optional[str] - the concern raised 
        all_bins: Optional[List[str]] - the list of bins
        bins_guidance_block : Optional[str]  — the list of bins and description as a block of string
        entry_id  : Optional[str]  — A unique identifier for this benchmark entry.
        metadata  : Dict[str, Any] — Additional metadata for the benchmark entry.
    """

    # ----- required -----
    bin: str
    comment: str
    binning_guidance_file: str

    # ----- optional -----
    concern: Optional[str] = None
    all_bins: Optional[List[str]] = None
    bins_guidance_block: Optional[str] = None
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
            self.entry_id = f"CBIN_{CommentBinBenchmarkEntry._id_counter}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bin": self.bin,
            "comment": self.comment,
            "concern": self.concern,
            "binning_guidance_file": self.binning_guidance_file,
            "all_bins": self.all_bins,
            "bins_guidance_block": self.bins_guidance_block,
            "entry_id": self.entry_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommentBinBenchmarkEntry':
        """
        Create an entry from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing entry data.
            
        Returns:
            CommentBinBenchmarkEntry: New instance created from the dictionary.
        """
        # Extract core fields
        core_fields = {k: v for k, v in data.items() 
                      if k in ['bin', 'comment', 'concern', 'binning_guidance_file', 
                               'all_bins', 'bins_guidance_block', 'entry_id'
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