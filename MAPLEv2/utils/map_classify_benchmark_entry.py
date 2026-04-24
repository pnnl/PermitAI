from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class MapClassifyBenchmarkEntry:
    """
    A benchmark entry for the Map-Classification tasks.

    Required Attributes:
        correct_class (str): the correct classification of the map image
        image_file (str): the file name of the image file

    Optional Attributes:
        classification_guidance_file (str): json file with classification guidance
        all_classes: Optional[List[str]] - the list of classes
        classification_guidance_block : Optional[str]  — the list of bins and description as a block of string
        entry_id  : Optional[str]  — A unique identifier for this benchmark entry.
        project: Optional[str] - project name for the image file
        project_summary: Optional[str] - project summary for the project
        pdf_name: Optional[str] - Name of the PDF file where the image is present
        metadata  : Dict[str, Any] — Additional metadata for the benchmark entry.
    """

    # ----- required -----
    correct_class: str
    image_file: str

    # ----- optional -----
    classification_guidance_file: Optional[str] = None
    all_classes: Optional[List[str]] = None
    classification_guidance_block: Optional[str] = None
    project: Optional[str] = None
    project_summary: Optional[str] = None
    pdf_name: Optional[str] = None
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
            self.entry_id = f"CBIN_{MapClassifyBenchmarkEntry._id_counter}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correct_class": self.correct_class,
            "image_file": self.image_file,
            "classification_guidance_file": self.classification_guidance_file,
            "all_classes": self.all_classes,
            "classification_guidance_block": self.classification_guidance_block,
            "project": self.project,
            "project_summary": self.project_summary,
            "pdf_name": self.pdf_name,
            "entry_id": self.entry_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MapClassifyBenchmarkEntry':
        """
        Create an entry from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing entry data.
            
        Returns:
            MapClassifyBenchmarkEntry: New instance created from the dictionary.
        """
        # Extract core fields
        core_fields = {k: v for k, v in data.items() 
                      if k in ['correct_class', 'image_file', 'classification_guidance_file', 
                               'all_classes', 'classification_guidance_block', 
                               'project', 'project_summary', 'pdf_name',
                               'entry_id'
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