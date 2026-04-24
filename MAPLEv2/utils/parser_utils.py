"""
Authors: 
    Kaustav Bhattacharya -- data parsing from document sections 
    Alex Buchko -- text/image parsing from documents
"""

import fitz
import pymupdf
import io
from typing import List

def is_next_section_valid(next_section: List[int], target_section: List[int]) -> bool:
    """
    Determines whether the next section in a table of contents is a valid subsequent section
    compared to a target section.

    Parameters:
        next_section (List[int]): The section numbers of the next section, extracted from a string format like "1.2.3".
        target_section (List[int]): The section numbers of the target section, extracted from a string format like "1.1".

    Returns:
        bool: True if the next section follows the target section logically, False otherwise.
    """
    if(len(next_section) == 0):
        return False
    if len(next_section) > len(target_section):
        # Compare all elements up to the length of target_section
        for i in range(len(target_section)):
            if next_section[i] > target_section[i]:
                return True
            elif next_section[i] < target_section[i]:
                return False
        # If all elements match, it's not valid (e.g., [3, 13, 1] vs [3, 13])
        return False
    
    # If lengths are equal or next_section is shorter
    for i in range(len(next_section)):
        if next_section[i] > target_section[i]:
            return True
        elif next_section[i] < target_section[i]:
            return False
    
    # If we've reached here, next_section is not valid
    return False

def extract_section_by_number(pdf_path: str, section_number: str) -> str:
    """
    Extracts text from a particular section in a PDF using its section number.

    Parameters:
        pdf_path (str): The path to the PDF file.
        section_number (str): The section number to extract, formatted as "1.2".

    Returns:
        str: The text content of the specified section, or "Section not found" if the target section cannot be located.
    """
    with fitz.open(pdf_path) as doc:
        toc = doc.get_toc()
        section_start = None
        section_end = None
        section_title = None
        next_section_title = None

        def parse_section(s):
            return [int(x) for x in s.split('.') if x.isdigit()]
        

        target_section = parse_section(section_number)

        for i, entry in enumerate(toc):
            #print(entry)
            if entry[1].startswith(section_number):
                section_start = entry[2] - 1  # Page numbers are 1-based
                section_title = entry[1]
                
                # Find the next higher section
                for j in range(i + 1, len(toc)):
                    next_entry = toc[j]
                    next_section = parse_section(next_entry[1].split()[0])
                    #print(next_entry,next_section, target_section, is_next_section_valid(next_section, target_section))
                    if is_next_section_valid(next_section, target_section):
                        section_end = next_entry[2] - 1
                        next_section_title = next_entry[1]
                        break
                
                # If no higher section is found, set end to the last page
                if section_end is None:
                    section_end = doc.page_count
                break

        #print(f"Section starts at page {section_start + 1}, ends at page {section_end + 1}")

        if section_start is not None:
            section_text = ""
            # Handling first page
            start_page = doc[section_start]
            start_page_text = start_page.get_text()
            section_name = section_title.split(' ', maxsplit=1)[1].strip()  # Get the section name after the number
            lines = start_page_text.split('\n')
            section_start_line_idx = -1
            for i, line in enumerate(lines):
                # Check for exact section number match at the beginning of a line
                if line.strip().startswith(section_number) and lines[i+1].strip() == section_name:
                    section_start_line_idx = i
                    break
            if section_start_line_idx >= 0:
                # If found, reassemble text starting from that line
                first_page_content = '\n'.join(lines[section_start_line_idx:])
            else:
                first_page_content = start_page_text    
            section_text += first_page_content
            
            
            # Handling middle pages
            if section_start != section_end:
                for page_num in range(section_start+1, section_end): # for multiple-page section
                    page = doc[page_num]
                    section_text += page.get_text()
            
            # Handling last page
            end_page = doc[section_end]
            end_page_text = end_page.get_text() 
            if(next_section_title):
                next_section_title_splits = next_section_title.split(' ', maxsplit=1)
                next_section_name = next_section_title_splits[1].strip()
                next_section_number = next_section_title_splits[0]
                lines = end_page_text.split('\n')
                section_end_line_idx = -1
                for i, line in enumerate(lines):
                    # Check for exact section number match at the beginning of a line
                    if line.strip().startswith(next_section_number) and lines[i+1].strip() == next_section_name:
                        section_end_line_idx = i
                        break
                if section_end_line_idx >= 0:
                    # If found, reassemble text starting from that line
                    last_page_content = '\n'.join(lines[:section_end_line_idx])
                else:
                    last_page_content = end_page_text
            section_text += last_page_content        
            return section_text
        else:
            return "Section not found"

def extract_section_by_name(pdf_path: str, section_name: str) -> str:
    """
    Extracts text from a particular section in a PDF using its section name.

    Parameters:
        pdf_path (str): The path to the PDF file.
        section_name (str): The name of the section to extract.

    Returns:
        str: The text content of the specified section, or "Section not found" if the target section cannot be located.
    """
    with fitz.open(pdf_path) as doc:
        toc = doc.get_toc()
        section_start = None
        section_end = None
        section_title = None
        next_section_title = None


        for i, entry in enumerate(toc):
            #print(entry)
            if (entry[1]).lower().startswith(section_name.lower()):
                section_start = entry[2] - 1  # Page numbers are 1-based
                main_level = entry[0]
                section_title = entry[1]
                
                # Find the next higher section
                for j in range(i + 1, len(toc)):
                    next_entry = toc[j]
                    next_level = next_entry[0]
                    #print(next_entry,next_section, target_section, is_next_section_valid(next_section, target_section))
                    #print(next_entry, next_level <= main_level)
                    if next_level <= main_level:
                        section_end = next_entry[2] - 1
                        next_section_title = next_entry[1]
                        break
                
                # If no higher section is found, set end to the last page
                if section_end is None:
                    section_end = doc.page_count
                break

        #print(f"Section starts at page {section_start + 1}, ends at page {section_end + 1}")

        if section_start is not None:
            section_text = ""
            # Handling first page
            start_page = doc[section_start]
            start_page_text = start_page.get_text()
            section_name_splits = section_title.split(' ', maxsplit=1)
            section_name = section_name_splits[1].strip()
            section_number = section_name_splits[0]
            lines = start_page_text.split('\n')
            section_start_line_idx = -1
            for i, line in enumerate(lines):
                # Check for exact section number match at the beginning of a line
                if line.strip().startswith(section_number) and lines[i+1].strip() == section_name:
                    section_start_line_idx = i
                    break
            if section_start_line_idx >= 0:
                # If found, reassemble text starting from that line
                first_page_content = '\n'.join(lines[section_start_line_idx:])
            else:
                first_page_content = start_page_text    
            section_text += first_page_content
            
            # Handling middle pages
            if section_start != section_end:
                for page_num in range(section_start+1, section_end): # for multiple-page section
                    page = doc[page_num]
                    section_text += page.get_text()
            
            # Handling last page
            end_page = doc[section_end]
            end_page_text = end_page.get_text() 
            if(next_section_title):
                if(next_section_title.startswith('Chapter')):
                    next_section_title_splits = next_section_title.split('-', maxsplit=1)
                    next_section_name = str(next_section_title_splits[1].strip()).upper() 
                    next_section_number = next_section_title_splits[0].strip()
                else:
                    next_section_title_splits = next_section_title.split(' ', maxsplit=1)
                    next_section_name = next_section_title_splits[1].strip() if len(next_section_title_splits) > 1 else next_section_title_splits[0]
                    next_section_number = next_section_title_splits[0]
                lines = end_page_text.split('\n')
                section_end_line_idx = -1
                for i, line in enumerate(lines):
                    # Check for exact section number match at the beginning of a line
                    if line.strip().startswith(next_section_number) and lines[i+1].strip() == next_section_name:
                        section_end_line_idx = i
                        break
                if section_end_line_idx >= 0:
                    # If found, reassemble text starting from that line
                    last_page_content = '\n'.join(lines[:section_end_line_idx])
                else:
                    last_page_content = end_page_text
            section_text += last_page_content  


            return section_text
        else:
            return "Section not found"

def parse_pdf(file_path):
    """
    Parse a PDF file and extract text from each page.
    
    :param file_path: Path to the PDF file
    :return: List of strings, each representing the text of a page
    """
    # Open the PDF file
    document = pymupdf.open(file_path)
    
    pages = []
    for page in document:
        #storing the text and images of each page
        page_data = {}
        page_data["images"] = []
        page_data["text"] = page.get_text()

        #getting the images of each page
        image_list = page.get_images()
        for image_index, img in enumerate(image_list):
            xref = img[0]  # get the XREF of the image
            # create a Pixmap
            pix =pymupdf.Pixmap(document, xref)
            pillow_image = pix.pil_image()
                
            # Convert the image to RGB if it's not already in that mode
            if pillow_image.mode != "RGB":
                pillow_image = pillow_image.convert("RGB")
            
            # Getting the image to bytes
            image_buffer = io.BytesIO()
            pillow_image.save(image_buffer, format="jpeg")
            image_bytes = image_buffer.getvalue()
            page_data["images"].append(image_bytes)

        pages.append(page_data)

    # Close the document
    document.close()
    
    return pages