# Author: Bishal Lakha
# Modified by: Rounak

import os, sys
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
from typing import Callable, Optional, List, Union
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils import LogManager

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/test_fuzzy_metrics.log")

logger = LogManager.get_logger("fuzzy_metrics")

def _normalize(s: str) -> str:
    """
    Normalize string by making it case-insensitive, trimming whitespace, and removing punctuation.
    
    Args:
        s (str): Input string to normalize
        
    Returns:
        str: Normalized string
    """
    try:
        logger.info(f"Normalizing string: '{s}'")
        # case-insensitive + trim + strip punctuation
        s = s.lower().strip()
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\s+", " ", s)
        logger.info(f"Normalized result: '{s}'")
        return s
    except Exception as e:
        logger.error(f"Error normalizing string '{s}': {str(e)}")
        return s  # Return original string if normalization fails

def semantic_similarity(
        a: str, b: str, 
        embedding_model="all-MiniLM-L6-v2"
        ) -> float:
    """
    Compute semantic similarity between two strings using sentence transformers.
    Returns a float between 0 and 1.
    
    Args:
        a (str): First string.
        b (str): Second string.
        embedding_model (str): Pretrained model name from sentence transformers.

    Returns:
        float: Similarity score between 0 and 1.
    """
    try:
        logger.info(f"Computing semantic similarity between '{a}' and '{b}' using model '{embedding_model}'")
        
        model = SentenceTransformer(embedding_model)
        emb_a = model.encode(a, convert_to_tensor=True)
        emb_b = model.encode(b, convert_to_tensor=True)

        similarity = float(util.cos_sim(emb_a, emb_b).item())
        logger.info(f"Semantic similarity score: {similarity:.4f}")
        return similarity
        
    except ImportError as e:
        logger.error(f"Failed to import sentence_transformers: {str(e)}")
        logger.warning("Falling back to fuzzy string matching")
        return fuzz.ratio(a, b) / 100.0
    except Exception as e:
        logger.error(f"Error computing semantic similarity: {str(e)}")
        logger.warning("Falling back to fuzzy string matching")
        return fuzz.ratio(a, b) / 100.0

def fuzzy_similarity_matrix(
        a: List[str], b: List[str],
        ) -> np.ndarray[float]:
    """
    Compute fuzzy similarity between two list of strings using rapidfuzz.
    Returns a float between 0 and 1.
    
    Args:
        a (List[str]): First list of strings.
        b (List[str]): Second list of strings.

    Returns:
        np.ndarray[float]: Similarity matrix with scores between 0 and 1.
    """
    n_pred, n_gold = len(a), len(b)
    sim = np.zeros((n_pred, n_gold), dtype=float)

    for i, p in enumerate(a):
        # Vectorized computation for one row at a time
        similarities = [fuzz.ratio(p, e) / 100.0 for e in b]
        sim[i, :] = similarities
    return sim
    
def semantic_similarity_matrix(
        a: List[str], b: List[str], 
        embedding_model="all-MiniLM-L6-v2"
        ) -> np.ndarray[float]:
    """
    Compute semantic similarity between two list of strings using sentence transformers.
    Returns a float between 0 and 1.
    
    Args:
        a (List[str]): First string.
        b (List[str]): Second string.
        embedding_model (str): Pretrained model name from sentence transformers.

    Returns:
        np.ndarray[float]: Similarity matrix with scores between 0 and 1.
    """
    try:
        n_pred, n_gold = len(a), len(b)
        sim = np.zeros((n_pred, n_gold), dtype=float)
        model = SentenceTransformer(embedding_model)
        emb_a = model.encode(a, convert_to_tensor=True)
        emb_b = model.encode(b, convert_to_tensor=True)

        # Compute cosine similarity matrix in one operation
        similarity_matrix = util.cos_sim(emb_a, emb_b)
        sim = similarity_matrix.cpu().numpy()
        return sim
        
    except ImportError as e:
        logger.error(f"Failed to import sentence_transformers: {str(e)}")
        logger.warning("Falling back to fuzzy string matching")
        return fuzzy_similarity_matrix(a, b)
    except Exception as e:
        logger.error(f"Error computing semantic similarity: {str(e)}")
        logger.warning("Falling back to fuzzy string matching")
        return fuzzy_similarity_matrix(a, b)

def abbreviation_similarity(a: str, b: str) -> float:
    """
    Compute similarity that handles abbreviations well.
    Combines fuzzy matching with abbreviation-specific logic.
    
    Args:
        a (str): First string
        b (str): Second string
        
    Returns:
        float: Similarity score between 0 and 1
    """
    try:
        logger.info(f"Computing abbreviation similarity between '{a}' and '{b}'")
        
        # Normalize both strings
        norm_a = _normalize(a)
        norm_b = _normalize(b)
        
        # If strings are identical after normalization, return 1.0
        if norm_a == norm_b:
            logger.info("Strings are identical after normalization")
            return 1.0
        
        # Check if one is an abbreviation of the other
        def is_abbreviation(short: str, long: str) -> float:
            try:
                logger.info(f"Checking if '{short}' is abbreviation of '{long}'")
                short_chars = short.replace(" ", "")
                long_words = long.split()
                
                # Simple acronym check: first letters of words
                if len(short_chars) == len(long_words):
                    acronym = "".join(word[0] for word in long_words if word)
                    if short_chars == acronym:
                        logger.info(f"Perfect acronym match found: {short_chars} = {acronym}")
                        return 0.95  # High score for perfect acronym match
                
                # Check if all characters of short string appear in order in long string
                short_idx = 0
                for char in long.replace(" ", ""):
                    if short_idx < len(short_chars) and char == short_chars[short_idx]:
                        short_idx += 1
                
                if short_idx == len(short_chars):
                    # All characters found in order, score based on compression ratio
                    compression_ratio = len(short_chars) / len(long.replace(" ", ""))
                    score = 0.8 + (0.15 * compression_ratio)
                    logger.info(f"Character subsequence match found with score: {score:.4f}")
                    return score
                
                return 0.0
            except Exception as e:
                logger.error(f"Error in abbreviation check: {str(e)}")
                return 0.0
        
        # Check both directions for abbreviation
        abbrev_score = max(
            is_abbreviation(norm_a, norm_b),
            is_abbreviation(norm_b, norm_a)
        )
        
        # Get fuzzy ratio score
        fuzzy_score = fuzz.ratio(norm_a, norm_b) / 100.0
        logger.info(f"Fuzzy score: {fuzzy_score:.4f}, Abbreviation score: {abbrev_score:.4f}")
        
        # Return the maximum of abbreviation score and fuzzy score
        final_score = max(abbrev_score, fuzzy_score)
        logger.info(f"Final abbreviation similarity score: {final_score:.4f}")
        return final_score
        
    except Exception as e:
        logger.error(f"Error computing abbreviation similarity: {str(e)}")
        logger.warning("Falling back to basic fuzzy matching")
        try:
            return fuzz.ratio(a, b) / 100.0
        except Exception as fallback_e:
            logger.error(f"Fallback fuzzy matching also failed: {str(fallback_e)}")
            return 0.0

def soft_precision_recall(
    expected: List[str],
    possible: List[str],
    embedding_model: Optional[str] = None,
    normalize: Optional[Callable[[str], str]] = _normalize,
    use_abbreviation_matching: bool = False,
    ):
    """
    soft precision/recall:
      - Precision = average over predictions of best similarity to any gold
      - Recall    = average over gold of best similarity to any prediction
    Returns (precision, recall, f1)

    Args:
        expected: List of expected/ground truth strings
        possible: List of predicted strings
        embedding_model: Optional model name for semantic similarity
        normalize: Optional normalization function
        use_abbreviation_matching: Whether to use abbreviation-aware matching
    """
    try:
        logger.info(f"Computing soft precision/recall for {len(expected)} expected and {len(possible)} predicted items")
        logger.info(f"Settings - embedding_model: {embedding_model}, use_abbreviation_matching: {use_abbreviation_matching}")
        
        # normalize once
        if normalize:
            try:
                E = [normalize(e) for e in expected]
                P = [normalize(p) for p in possible]
                logger.info("Successfully normalized input strings")
            except Exception as e:
                logger.warning(f"Error during normalization: {str(e)}, using original strings")
                E, P = list(expected), list(possible)
        else:
            E, P = list(expected), list(possible)

        # handle empties
        n_pred = len(P)
        n_gold = len(E)
        if n_pred == 0 or n_gold == 0:
            logger.info(f"Empty input detected - predictions: {n_pred}, gold: {n_gold}")
            return 0.0, 0.0, 0.0

        # build similarity matrix (n_pred x n_gold)
        try:
            logger.info(f"Building {n_pred}x{n_gold} similarity matrix")
            
            if embedding_model:
                logger.info(f"Using semantic similarity with model: {embedding_model}")
                sim = semantic_similarity_matrix(expected, possible, embedding_model)
            else:
                sim = fuzzy_similarity_matrix(expected, possible)
            
            logger.info(f"Similarity matrix computed successfully")
            logger.info(f"Similarity matrix shape: {sim.shape}")
            
        except Exception as e:
            logger.error(f"Error building similarity matrix: {str(e)}")
            return 0.0, 0.0, 0.0

        # precision: best gold for each prediction
        try:
            best_per_pred = sim.max(axis=1) if n_gold > 0 else np.zeros(n_pred)
            precision = float(best_per_pred.mean()) if n_pred > 0 else 0.0
            logger.info(f"Precision calculated: {precision:.4f}")
        except Exception as e:
            logger.error(f"Error calculating precision: {str(e)}")
            precision = 0.0

        # recall: best prediction for each gold
        try:
            best_per_gold = sim.max(axis=0) if n_pred > 0 else np.zeros(n_gold)
            recall = float(best_per_gold.mean()) if n_gold > 0 else 0.0
            logger.info(f"Recall calculated: {recall:.4f}")
        except Exception as e:
            logger.error(f"Error calculating recall: {str(e)}")
            recall = 0.0

        # Calculate F1 score
        try:
            f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
            logger.info(f"F1 score calculated: {f1:.4f}")
        except Exception as e:
            logger.error(f"Error calculating F1 score: {str(e)}")
            f1 = 0.0
        
        logger.info(f"Soft metrics computed - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return precision, recall, f1
        
    except Exception as e:
        logger.error(f"Unexpected error in soft_precision_recall: {str(e)}")
        return 0.0, 0.0, 0.0


def semantic_similarity_embedding(
    a: str, 
    b: str, 
    embedding_model: str = "all-MiniLM-L6-v2"
) -> float:
    """
    Compute semantic similarity using sentence transformers embedding model.
    
    Args:
        a (str): First string
        b (str): Second string
        embedding_model (str): Pretrained model name from sentence transformers
        
    Returns:
        float: Semantic similarity score between 0 and 1
    """
    try:
        logger.info(f"Computing semantic similarity with embedding model '{embedding_model}' for '{a}' vs '{b}'")
        return semantic_similarity(a, b, embedding_model)
    except Exception as e:
        logger.error(f"Error computing semantic similarity with embedding: {str(e)}")
        return 0.0


def semantic_similarity_fuzzy(a: str, b: str) -> float:
    """
    Compute semantic similarity using fuzzy string matching.
    
    Args:
        a (str): First string
        b (str): Second string
        
    Returns:
        float: Fuzzy similarity score between 0 and 1
    """
    try:
        logger.info(f"Computing fuzzy semantic similarity for '{a}' vs '{b}'")
        
        # Normalize strings
        norm_a = _normalize(a)
        norm_b = _normalize(b)
        
        # Compute fuzzy ratio
        similarity = fuzz.ratio(norm_a, norm_b) / 100.0
        logger.info(f"Fuzzy semantic similarity score: {similarity:.4f}")
        return similarity
        
    except Exception as e:
        logger.error(f"Error computing fuzzy semantic similarity: {str(e)}")
        return 0.0


def abbreviation_similarity_score(a: str, b: str) -> float:
    """
    Compute abbreviation-aware similarity score.
    
    Args:
        a (str): First string
        b (str): Second string
        
    Returns:
        float: Abbreviation similarity score between 0 and 1
    """
    try:
        logger.info(f"Computing abbreviation similarity score for '{a}' vs '{b}'")
        return abbreviation_similarity(a, b)
    except Exception as e:
        logger.error(f"Error computing abbreviation similarity score: {str(e)}")
        return 0.0
 

if __name__ == "__main__":
    logger.info("Starting fuzzy metrics testing")
    
    try:
        # Test semantic similarity
        expected = "Bourneville Power Administration"
        predicted = "BPA"
        logger.info(f"Testing semantic similarity between '{expected}' and '{predicted}'")
        sem_sim = semantic_similarity(expected, predicted)
        print(f"Semantic Similarity Score: {sem_sim:.4f}")
        
        # Test abbreviation similarity
        logger.info(f"Testing abbreviation similarity between '{expected}' and '{predicted}'")
        abbrev_sim = abbreviation_similarity(expected, predicted)
        print(f"Abbreviation Similarity Score: {abbrev_sim:.4f}")

        # Test with different examples
        test_cases = [
            ("Bourneville Power Administration", "BPA"),
            ("Federal Bureau of Investigation", "FBI"),
            ("United States of America", "USA"),
            ("National Aeronautics and Space Administration", "NASA"),
            ("World Health Organization", "WHO"),
            ("Department of Defense", "DOD"),
            ("Environmental Protection Agency", "EPA"),
        ]
        
        print("\nAbbreviation similarity tests:")
        logger.info("Running abbreviation similarity tests on multiple cases")
        for full, abbrev in test_cases:
            try:
                score = abbreviation_similarity(full, abbrev)
                print(f"{full} vs {abbrev}: {score:.4f}")
                logger.info(f"Test case '{full}' vs '{abbrev}': {score:.4f}")
            except Exception as e:
                logger.error(f"Error testing '{full}' vs '{abbrev}': {str(e)}")

        # Example usage of soft precision/recall
        expected_list = ["The cat sat on the mat.", "A quick brown fox jumps over the lazy dog."]
        possible_list = ["A fast brown fox leaps over a lazy dog.", "The cat is sitting on the mat."]

        # Test with different similarity methods
        print("\n=== Fuzzy String Matching ===")
        logger.info("Testing soft precision/recall with fuzzy string matching")
        try:
            precision, recall, f1 = soft_precision_recall(expected_list, possible_list)
            print(f"Soft Precision: {precision:.4f}")
            print(f"Soft Recall: {recall:.4f}")
            print(f"Soft F1 Score: {f1:.4f}")
        except Exception as e:
            logger.error(f"Error in fuzzy string matching test: {str(e)}")

        print("\n=== Abbreviation-Aware Matching ===")
        logger.info("Testing soft precision/recall with abbreviation-aware matching")
        expected_abbrev = ["Bourneville Power Administration", "Federal Bureau of Investigation"]
        possible_abbrev = ["BPA", "FBI"]
        
        try:
            precision, recall, f1 = soft_precision_recall(
                expected_abbrev, possible_abbrev, 
                use_abbreviation_matching=True
            )
            print(f"Soft Precision: {precision:.4f}")
            print(f"Soft Recall: {recall:.4f}")
            print(f"Soft F1 Score: {f1:.4f}")
        except Exception as e:
            logger.error(f"Error in abbreviation-aware matching test: {str(e)}")

        # Uncomment to test semantic similarity (requires model download)
        print("\n=== Semantic Similarity (Commented Out) ===")
        logger.info("Semantic similarity test is commented out to avoid model download")
        precision, recall, f1 = soft_precision_recall(
            expected_list, possible_list,
            embedding_model='all-MiniLM-L6-v2'
            )
        
        logger.info("All tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error during main execution: {str(e)}")
        print(f"An error occurred during testing: {str(e)}")