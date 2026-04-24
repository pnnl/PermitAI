import os
import sys
import re
from typing import Dict, List, Any, Union

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_prompt_manager.log")

logger = LogManager.get_logger(__name__)

from utils import (
    QABenchmarkEntry, 
    IEBenchmarkEntry, TribalBenchmarkEntry, StructuredIEBenchmarkEntry, 
    CommentBinBenchmarkEntry, CommentClassificationBenchmarkEntry, BinSummarizerBenchmarkEntry, 
    MapClassifyBenchmarkEntry, CommentDelineateBenchmarkEntry
)

class PromptManager:
    """
    Manages loading and formatting prompts from text files.
    
    This class works exclusively with QABenchmarkEntry, IEBenchmarkEntry, TribalBenchmarkEntry, 
    StructuredIEBenchmarkEntry, CommentBinBenchmarkEntry, BinSummarizerBenchmarkEntry 
    and MapClassifyBenchmarkEntry objects for prompt formatting, 
    replacing placeholders with values from these benchmark entries.
    """
    
    def __init__(self, template_source: str, is_file_path: bool = True):
        """
        Initialize the PromptManager with a prompt template from file or string.
        
        Args:
            template_source (str): Either a path to the template file or the template string itself.
            is_file_path (bool): If True, template_source is treated as a file path. 
                                If False, template_source is treated as the template string.
        """
        logger.info(f"Initializing PromptManager with template {'from file' if is_file_path else 'from string'}")
        self.template_source = template_source
        self.is_file_path = is_file_path
        
        if is_file_path:
            self.template_path = template_source
            self.template = self.load_prompt_template(template_source)
        else:
            self.template_path = None
            self.template = template_source
            logger.info(f"Using template string of {len(self.template)} characters")
            
        self.prompt = None  # Will hold the most recently formatted prompt
        
        # Split the template into parts and extract placeholders
        self.parts = self._split_template_by_code_blocks(self.template)
        self.placeholders = self._find_placeholders(self.parts)
        
        logger.info(f"Extracted {len(self.placeholders)} placeholders: {', '.join(self.placeholders)}")
    
    @classmethod
    def from_file(cls, template_path: str):
        """
        Create a PromptManager from a template file.
        
        Args:
            template_path (str): Path to the template file.
            
        Returns:
            PromptManager: Instance initialized with the template file.
        """
        return cls(template_path, is_file_path=True)
    
    @classmethod
    def from_string(cls, template_string: str):
        """
        Create a PromptManager from a template string.
        
        Args:
            template_string (str): The template string containing placeholders.
            
        Returns:
            PromptManager: Instance initialized with the template string.
        """
        return cls(template_string, is_file_path=False)

    def _split_template_by_code_blocks(self, template: str) -> List[tuple]:
        """
        Split a template into code blocks and non-code blocks.
        
        Args:
            template (str): The template string.
            
        Returns:
            List[tuple]: List of (text, is_code_block) tuples.
        """
        parts = []
        is_code_block = False
        current_part = ""
        
        for line in template.split('\n'):
            if line.strip().startswith('```'):
                # Process the current part if it's not empty
                if current_part:
                    parts.append((current_part, is_code_block))
                    current_part = ""
                
                # Add the code block marker
                current_part += line + '\n'
                
                # Toggle code block state
                is_code_block = not is_code_block
            else:
                current_part += line + '\n'
        
        # Add the last part if not empty
        if current_part:
            parts.append((current_part, is_code_block))
            
        return parts
    
    def _find_placeholders(self, parts: List[tuple]) -> List[str]:
        """
        Find unique placeholders in non-code parts of the template.
        
        Args:
            parts (List[tuple]): List of (text, is_code_block) tuples.
            
        Returns:
            List[str]: List of unique placeholder names.
        """
        placeholders = []
        for part, is_code in parts:
            if not is_code:
                placeholders.extend(re.findall(r'\{(\w+)\}', part))
        
        # Remove duplicates
        return list(set(placeholders))
    
    def _validate_placeholders(self, placeholders: List[str], entry_dict: Dict[str, Any]) -> None:
        """
        Validate that all placeholders exist in the entry dictionary.
        Check metadata dictionary if available.
        
        Args:
            placeholders (List[str]): List of placeholder names.
            entry_dict (Dict[str, Any]): Dictionary of values.
            
        Raises:
            ValueError: If a placeholder is not found in the entry dictionary.
        """
        for placeholder in placeholders:
            # Check in main dictionary
            if placeholder in entry_dict:
                continue
                
            # Check in metadata if it exists
            if 'metadata' in entry_dict and isinstance(entry_dict['metadata'], dict):
                if placeholder in entry_dict['metadata']:
                    continue
            
            error_msg = f"Placeholder '{placeholder}' not found in benchmark entry fields or metadata"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _format_parts(self, parts: List[tuple], entry_dict: Dict[str, Any]) -> str:
        """
        Format non-code parts and leave code parts unchanged.
        Handle metadata fields for new entry types.
        
        Args:
            parts (List[tuple]): List of (text, is_code_block) tuples.
            entry_dict (Dict[str, Any]): Dictionary of values.
            
        Returns:
            str: The formatted prompt.
        """
        formatted_parts = []
        format_dict = dict(entry_dict)  # Create a copy to avoid modifying the original
        
        # If there's metadata, flatten it into the format_dict for easier access
        if 'metadata' in format_dict and isinstance(format_dict['metadata'], dict):
            for key, value in format_dict['metadata'].items():
                # Only add if the key doesn't already exist in the main dict
                if key not in format_dict:
                    format_dict[key] = value
        
        for part, is_code in parts:
            if is_code:
                formatted_parts.append(part)
            else:
                formatted_parts.append(part.format(**format_dict))
        
        return ''.join(formatted_parts)
    
    def load_prompt_template(self, template_path: str) -> str:
        """
        Load a prompt template from a text file.
        
        Args:
            template_path (str): Path to the template file.
            
        Returns:
            str: The prompt template as a string.
        """
        logger.info(f"Loading prompt template from {template_path}")
        
        try:
            with open(template_path, 'r', encoding='utf-8') as file:
                template = file.read()
            logger.info(f"Successfully loaded prompt template of {len(template)} characters")
            return template
        except Exception as e:
            logger.error(f"Error loading prompt template: {str(e)}")
            raise
    
    def format_prompt(self, entry) -> str:
        """
        Format the prompt template by replacing placeholders with values from a benchmark entry.
        Stores the formatted prompt as self.prompt.
        
        Args:
            entry: The benchmark entry containing the values to substitute.
                  Must be either QABenchmarkEntry, IEBenchmarkEntry, TribalBenchmarkEntry or StructuredIEBenchmarkEntry.
            
        Returns:
            str: The formatted prompt.
        """
        try:
            # Validate entry type
            if not isinstance(
                entry, 
                (QABenchmarkEntry, IEBenchmarkEntry, TribalBenchmarkEntry, 
                 StructuredIEBenchmarkEntry, CommentBinBenchmarkEntry, 
                 CommentClassificationBenchmarkEntry, BinSummarizerBenchmarkEntry, 
                 MapClassifyBenchmarkEntry,CommentDelineateBenchmarkEntry)
                ):
                error_msg = f"Unsupported entry type: {type(entry)}. Must be QABenchmarkEntry, IEBenchmarkEntry, TribalBenchmarkEntry, StructuredIEBenchmarkEntry, CommentBinBenchmarkEntry, CommentClassificationBenchmarkEntry, BinSummarizerBenchmarkEntry, MapClassifyBenchmarkEntry."
                logger.error(error_msg)
                raise TypeError(error_msg)
            
            # Convert entry to dictionary
            entry_dict = entry.to_dict()
            
            # Check if all placeholders exist in the entry
            self._validate_placeholders(self.placeholders, entry_dict)
            
            # Format each non-code part and leave code parts unchanged
            self.prompt = self._format_parts(self.parts, entry_dict)
            
            logger.info(f"Successfully formatted prompt with {len(self.placeholders)} placeholders")
            
            return self.prompt
            
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            raise
    
    def get_placeholder_info(self) -> Dict[str, Any]:
        """
        Get information about the placeholders in the prompt.
        
        Returns:
            Dict[str, Any]: Dictionary with placeholder information
        """
        info = {
            "placeholders": self.placeholders,
            "placeholder_count": len(self.placeholders),
            "has_formatted_prompt": self.prompt is not None,
            "is_file_based": self.is_file_path
        }
        
        if self.is_file_path:
            info["template_path"] = self.template_path
        else:
            info["template_length"] = len(self.template)
            
        return info
    
    def get_formatted_prompt(self) -> str:
        """
        Get the most recently formatted prompt.
        
        Returns:
            str: The formatted prompt or None if no prompt has been formatted yet.
            
        Raises:
            ValueError: If no prompt has been formatted yet.
        """
        if self.prompt is None:
            raise ValueError("No prompt has been formatted yet. Call format_prompt() first.")
        return self.prompt
    
    def get_template(self) -> str:
        """
        Get the original template string.
        
        Returns:
            str: The original template string with placeholders.
        """
        return self.template
