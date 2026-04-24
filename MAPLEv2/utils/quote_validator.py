import re
from difflib import SequenceMatcher
from typing import List, Dict

def validate_quotes_in_text(quotes: List[str], full_text: str, 
                           min_overlap_threshold: float = 0.8,
                           min_consecutive_words: int = 5) -> Dict:
    """
    Validate if generated quotes are present in the full text.
    Returns only quotes that are entirely or substantially present.
    
    Args:
        quotes (List[str]): List of quotes to validate
        full_text (str): The source text to search within
        min_overlap_threshold (float): Minimum overlap ratio to consider valid (0.0-1.0)
        min_consecutive_words (int): Minimum consecutive words required for partial matches
    
    Returns:
        Dict: Validation results with valid/invalid quotes and details
    """
    # Clean and normalize text for comparison
    def clean_text(text: str) -> str:
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def normalize_for_comparison(text: str) -> str:
        # Convert to lowercase and remove punctuation for fuzzy matching
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    # Prepare normalized full text
    full_text_clean = clean_text(full_text)
    full_text_normalized = normalize_for_comparison(full_text_clean)
    full_text_words = full_text_normalized.split()
    
    valid_quotes = []
    invalid_quotes = []
    validation_details = []
    
    for i, quote in enumerate(quotes):
        if not quote.strip():
            invalid_quotes.append(quote)
            validation_details.append({
                'quote_index': i,
                'quote': quote,
                'status': 'invalid',
                'reason': 'empty_quote',
                'overlap_ratio': 0.0,
                'match_type': None
            })
            continue
        
        quote_clean = clean_text(quote)
        quote_normalized = normalize_for_comparison(quote_clean)
        quote_words = quote_normalized.split()
        
        # Check for exact match first
        if quote_clean.lower() in full_text_clean.lower():
            valid_quotes.append(quote)
            validation_details.append({
                'quote_index': i,
                'quote': quote,
                'status': 'valid',
                'reason': 'exact_match',
                'overlap_ratio': 1.0,
                'match_type': 'exact'
            })
            continue
        
        # Check for high similarity using sequence matching
        matcher = SequenceMatcher(None, quote_normalized, full_text_normalized)
        similarity_ratio = matcher.ratio()
        
        # Find longest common subsequence
        matching_blocks = matcher.get_matching_blocks()
        longest_match = max(matching_blocks, key=lambda x: x.size) if matching_blocks else None
        
        # Check for substantial word overlap
        quote_words_set = set(quote_words)
        full_text_words_set = set(full_text_words)
        overlapping_words = quote_words_set.intersection(full_text_words_set)
        word_overlap_ratio = len(overlapping_words) / len(quote_words) if quote_words else 0
        
        # Check for consecutive word sequences
        max_consecutive = find_max_consecutive_words(quote_words, full_text_words)
        
        # Determine if quote is valid based on multiple criteria
        is_valid = False
        match_type = None
        reason = None
        
        if word_overlap_ratio >= min_overlap_threshold:
            is_valid = True
            match_type = 'high_word_overlap'
            reason = f'word_overlap_{word_overlap_ratio:.2f}'
        elif max_consecutive >= min_consecutive_words:
            is_valid = True
            match_type = 'consecutive_sequence'
            reason = f'consecutive_words_{max_consecutive}'
        elif similarity_ratio >= 0.7:  # High similarity threshold
            is_valid = True
            match_type = 'high_similarity'
            reason = f'similarity_{similarity_ratio:.2f}'
        else:
            is_valid = False
            match_type = 'insufficient_match'
            reason = f'low_overlap_{word_overlap_ratio:.2f}_similarity_{similarity_ratio:.2f}'
        
        if is_valid:
            valid_quotes.append(quote)
        else:
            invalid_quotes.append(quote)
        
        validation_details.append({
            'quote_index': i,
            'quote': quote,
            'status': 'valid' if is_valid else 'invalid',
            'reason': reason,
            'overlap_ratio': word_overlap_ratio,
            'similarity_ratio': similarity_ratio,
            'max_consecutive_words': max_consecutive,
            'match_type': match_type
        })
    
    return {
        'valid_quotes': valid_quotes,
        'invalid_quotes': invalid_quotes,
        'validation_details': validation_details,
        'summary': {
            'total_quotes': len(quotes),
            'valid_count': len(valid_quotes),
            'invalid_count': len(invalid_quotes),
            'validity_rate': len(valid_quotes) / len(quotes) if quotes else 0
        }
    }

def find_max_consecutive_words(quote_words: List[str], full_text_words: List[str]) -> int:
    """Find the maximum number of consecutive words from quote that appear in full text."""
    max_consecutive = 0
    
    for i in range(len(quote_words)):
        for j in range(len(full_text_words)):
            consecutive_count = 0
            
            # Check consecutive matches starting from positions i and j
            while (i + consecutive_count < len(quote_words) and 
                   j + consecutive_count < len(full_text_words) and
                   quote_words[i + consecutive_count] == full_text_words[j + consecutive_count]):
                consecutive_count += 1
            
            max_consecutive = max(max_consecutive, consecutive_count)
    
    return max_consecutive

def filter_valid_quotes(quotes: List[str], full_text: str, 
                       min_overlap_threshold: float = 0.8) -> List[str]:
    """
    Simple function to return only valid quotes.
    
    Args:
        quotes (List[str]): List of quotes to validate
        full_text (str): The source text to search within
        min_overlap_threshold (float): Minimum overlap ratio to consider valid
    
    Returns:
        List[str]: Only the quotes that are valid
    """
    validation_result = validate_quotes_in_text(quotes, full_text, min_overlap_threshold)
    return validation_result['valid_quotes']

def sample_data():
    # Test with sample data
    full_text = """
    The Surety & Fidelity Association of America appreciates the opportunity to provide written 
    comments related to the Bureau of Land Management's proposed Fluid Mineral Leases and Leasing 
    Process Rule. SFAA commends BLM for updating the minimum bond amounts which were last updated 
    in 1951 and 1960, respectively, to more accurately reflect the likely cost of the operator's 
    plugging, reclamation, and restoration obligations. The proposed rule should specify that the 
    required bond amounts for lease bonds covering more than two wells must increase by a given 
    amount per well. BLM should adopt an accountability program similar to the Applicant Violator 
    System administered by the Office of Surface Mining.
    """
    
    test_quotes = [
        # Valid - exact match
        "SFAA commends BLM for updating the minimum bond amounts which were last updated in 1951 and 1960, respectively",
        
        # Valid - high overlap with minor differences
        "The proposed rule should specify that required bond amounts for lease bonds covering more than two wells must increase",
        
        # Valid - partial but substantial match
        "BLM should adopt an accountability program similar to the Applicant Violator System",
        
        # Invalid - completely different content
        "The environmental impact assessment showed significant concerns about wildlife habitat",
        
        # Invalid - minimal overlap
        "Companies should prioritize renewable energy investments over fossil fuel development"
    ]
    return full_text, test_quotes

def log_results(full_text, test_quotes):
    output_text = []
    output_text.append("=== Quote Validation Test ===")
    result = validate_quotes_in_text(test_quotes, full_text)
    
    output_text.append(f"\nSummary:")
    output_text.append(f"Total quotes: {result['summary']['total_quotes']}")
    output_text.append(f"Valid quotes: {result['summary']['valid_count']}")
    output_text.append(f"Invalid quotes: {result['summary']['invalid_count']}")
    output_text.append(f"Validity rate: {result['summary']['validity_rate']:.2%}")
    
    output_text.append(f"\nValid Quotes:")
    for quote in result['valid_quotes']:
        output_text.append(f"✓ {quote[:100]}...")
    
    output_text.append(f"\nInvalid Quotes:")
    for quote in result['invalid_quotes']:
        output_text.append(f"✗ {quote[:100]}...")
    
    output_text.append(f"\nDetailed Analysis:")
    for detail in result['validation_details']:
        output_text.append(f"Quote {detail['quote_index']}: {detail['status']} - {detail['reason']} "
              f"(overlap: {detail['overlap_ratio']:.2f})")
    
    output_text.append(f"\n=== Simple Filter Test ===")
    valid_only = filter_valid_quotes(test_quotes, full_text)
    output_text.append(f"Filtered to {len(valid_only)} valid quotes out of {len(test_quotes)} total")
    
    return '\n'.join(output_text)
    
    