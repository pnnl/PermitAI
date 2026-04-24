import io
import mimetypes
import base64
import os
import sys
import tempfile
import uuid
from PIL import Image
from pathlib import Path

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_image_utils.log")

logger = LogManager.get_logger("image_utils")

def _get_temp_image_path(original_path: str, suffix: str = "_processed") -> str:
    """
    Generate a temporary file path for processed images.
    
    Args:
        original_path (str): Path to the original image.
        suffix (str): Suffix to add to the filename.
        
    Returns:
        str: Path to the temporary processed image file.
    """
    original_path = Path(original_path)
    temp_dir = tempfile.gettempdir()
    unique_id = str(uuid.uuid4())[:8]
    
    # Create filename: original_name + suffix + unique_id + .jpg
    temp_filename = f"{original_path.stem}{suffix}_{unique_id}.jpg"
    temp_path = Path(temp_dir) / temp_filename
    
    return str(temp_path)

def _resize_image_for_claude(image_path: str, max_dimension: int = 7500) -> tuple:
    """
    Resize image to fit Claude's size constraints while maintaining aspect ratio.
    Saves to a temporary file without overwriting the original.
    
    Args:
        image_path (str): Path to the original image file.
        max_dimension (int): Maximum allowed dimension in pixels.
        
    Returns:
        tuple: (base64_encoded_string, temp_file_path)
        
    Raises:
        Exception: If image processing fails.
    """
    temp_path = None
    try:
        with Image.open(image_path) as img:
            # Get current dimensions
            width, height = img.size
            logger.info(f"Original image dimensions: {width}x{height}")
            
            # Check if resizing is needed
            if width <= max_dimension and height <= max_dimension:
                logger.info("Image dimensions within limits, no resize needed")
                # Return original without creating temp file
                with open(image_path, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8'), None
            
            # Calculate new dimensions maintaining aspect ratio
            if width > height:
                new_width = max_dimension
                new_height = int((height * max_dimension) / width)
            else:
                new_height = max_dimension
                new_width = int((width * max_dimension) / height)
            
            logger.info(f"Resizing image to: {new_width}x{new_height}")
            
            # Resize image with high quality resampling
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary for JPEG saving
            if resized_img.mode in ['RGBA', 'P']:
                background = Image.new('RGB', resized_img.size, (255, 255, 255))
                if resized_img.mode == 'P':
                    resized_img = resized_img.convert('RGBA')
                if resized_img.mode == 'RGBA':
                    background.paste(resized_img, mask=resized_img.split()[-1])
                resized_img = background
            
            # Save to temporary file
            temp_path = _get_temp_image_path(image_path, "_resized")
            resized_img.save(temp_path, format='JPEG', quality=85, optimize=True)
            
            # Log the size reduction
            original_size = os.path.getsize(image_path)
            new_size = os.path.getsize(temp_path)
            logger.info(f"Image size reduced from {original_size} to {new_size} bytes "
                       f"({(new_size/original_size)*100:.1f}% of original)")
            logger.info(f"Temporary resized image saved to: {temp_path}")
            
            # Encode the temporary file
            with open(temp_path, "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode('utf-8')
            
            return encoded_string, temp_path
            
    except Exception as e:
        # Clean up temp file if it was created
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Cleaned up temporary file after error: {temp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temp file {temp_path}: {cleanup_error}")
        
        logger.error(f"Error resizing image {image_path}: {str(e)}")
        raise


def _compress_image_for_claude(image_path: str, target_size_mb: float = 4.5, min_quality: int = 20) -> tuple:
    """
    Compress image to meet Claude's file size requirements while maintaining acceptable quality.
    Saves to a temporary file without overwriting the original.
    
    Args:
        image_path (str): Path to the original image file.
        target_size_mb (float): Target file size in MB.
        min_quality (int): Minimum quality threshold (1-100).
        
    Returns:
        tuple: (base64_encoded_string, temp_file_path)
        
    Raises:
        Exception: If image cannot be compressed to target size.
    """
    temp_path = None
    try:
        with Image.open(image_path) as img:
            original_size = os.path.getsize(image_path)
            original_size_mb = original_size / (1024 * 1024)
            
            # Check if compression is needed
            if original_size_mb <= target_size_mb:
                logger.info(f"Image size ({original_size_mb:.2f}MB) already within target ({target_size_mb}MB)")
                with open(image_path, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8'), None
            
            logger.info(f"Compressing image from {original_size_mb:.2f}MB to target {target_size_mb}MB")
            
            # Convert to RGB if necessary (for JPEG compression)
            if img.mode in ['RGBA', 'P']:
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[-1])
                img = background
            
            # Try different compression levels
            target_size_bytes = target_size_mb * 1024 * 1024
            quality = 95
            temp_path = _get_temp_image_path(image_path, "_compressed")
            
            while quality >= min_quality:
                img.save(temp_path, format='JPEG', quality=quality, optimize=True)
                current_size = os.path.getsize(temp_path)
                
                if current_size <= target_size_bytes:
                    current_size_mb = current_size / (1024 * 1024)
                    compression_ratio = (current_size / original_size) * 100
                    
                    logger.info(f"Compressed to {current_size_mb:.2f}MB (quality={quality}, "
                               f"{compression_ratio:.1f}% of original)")
                    logger.info(f"Temporary compressed image saved to: {temp_path}")
                    
                    # Encode the compressed file
                    with open(temp_path, "rb") as f:
                        encoded_string = base64.b64encode(f.read()).decode('utf-8')
                    
                    return encoded_string, temp_path
                
                quality -= 5  # Reduce quality in steps
            
            # If still too large, try with dimension reduction
            logger.warning(f"Could not compress to target size with quality >= {min_quality}")
            os.remove(temp_path)  # Remove the failed attempt
            return _compress_with_dimension_reduction_temp(img, image_path, target_size_mb, min_quality)
            
    except Exception as e:
        # Clean up temp file if it was created
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Cleaned up temporary file after error: {temp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temp file {temp_path}: {cleanup_error}")
        
        logger.error(f"Error compressing image {image_path}: {str(e)}")
        raise


def _compress_with_dimension_reduction_temp(img: Image.Image, original_path: str, 
                                          target_size_mb: float, min_quality: int = 30) -> tuple:
    """
    Compress image by reducing both dimensions and quality, saving to temp file.
    
    Args:
        img (Image.Image): PIL Image object.
        original_path (str): Path to original image file.
        target_size_mb (float): Target file size in MB.
        min_quality (int): Minimum quality threshold.
        
    Returns:
        tuple: (base64_encoded_string, temp_file_path)
    """
    temp_path = None
    try:
        target_size_bytes = target_size_mb * 1024 * 1024
        original_width, original_height = img.size
        temp_path = _get_temp_image_path(original_path, "_dim_reduced")
        
        # Try different scale factors
        scale_factors = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        
        for scale in scale_factors:
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Don't go below reasonable minimum dimensions
            if new_width < 200 or new_height < 200:
                continue
                
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Try different qualities for this size
            for quality in range(85, min_quality - 1, -10):
                resized_img.save(temp_path, format='JPEG', quality=quality, optimize=True)
                current_size = os.path.getsize(temp_path)
                
                if current_size <= target_size_bytes:
                    current_size_mb = current_size / (1024 * 1024)
                    logger.info(f"Compressed with dimension reduction: {new_width}x{new_height}, "
                               f"quality={quality}, size={current_size_mb:.2f}MB")
                    logger.info(f"Temporary optimized image saved to: {temp_path}")
                    
                    # Encode the optimized file
                    with open(temp_path, "rb") as f:
                        encoded_string = base64.b64encode(f.read()).decode('utf-8')
                    
                    return encoded_string, temp_path
        
        # Clean up and raise exception if we couldn't optimize enough
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        raise Exception(f"Could not compress image to target size {target_size_mb}MB")
        
    except Exception as e:
        # Clean up temp file if it was created
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise


def smart_image_optimization(image_path: str, max_dimension: int = 7500, target_size_mb: float = 4.5) -> tuple:
    """
    Intelligently optimize image for Claude by handling both dimensions and file size.
    Saves optimized image to temporary file.
    
    Args:
        image_path (str): Path to the original image file.
        max_dimension (int): Maximum allowed dimension in pixels.
        target_size_mb (float): Target file size in MB.
        
    Returns:
        tuple: (base64_encoded_string, temp_file_path)
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            file_size = os.path.getsize(image_path)
            file_size_mb = file_size / (1024 * 1024)
            
            logger.info(f"Optimizing image: {width}x{height}, {file_size_mb:.2f}MB")
            
            needs_dimension_resize = width > max_dimension or height > max_dimension
            needs_compression = file_size_mb > target_size_mb
            
            if not needs_dimension_resize and not needs_compression:
                # No optimization needed
                logger.info("Image already optimized, no changes needed")
                with open(image_path, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8'), None
            
            # Strategy 1: If only dimensions are the problem
            if needs_dimension_resize and not needs_compression:
                logger.info("Resizing dimensions only")
                return _resize_image_for_claude(image_path, max_dimension)
            
            # Strategy 2: If only file size is the problem
            elif needs_compression and not needs_dimension_resize:
                logger.info("Compressing file size only")
                return _compress_image_for_claude(image_path, target_size_mb)
            
            # Strategy 3: Both dimensions and file size need optimization
            else:
                logger.info("Optimizing both dimensions and file size")
                return _comprehensive_optimization_temp(img, image_path, max_dimension, target_size_mb)
                
    except Exception as e:
        logger.error(f"Error in smart image optimization for {image_path}: {str(e)}")
        raise


def _comprehensive_optimization_temp(img: Image.Image, original_path: str, 
                                   max_dimension: int, target_size_mb: float) -> tuple:
    """
    Perform comprehensive optimization for both dimensions and file size, saving to temp file.
    
    Args:
        img (Image.Image): PIL Image object.
        original_path (str): Path to original image file.
        max_dimension (int): Maximum allowed dimension.
        target_size_mb (float): Target file size in MB.
        
    Returns:
        tuple: (base64_encoded_string, temp_file_path)
    """
    temp_path = None
    try:
        target_size_bytes = target_size_mb * 1024 * 1024
        width, height = img.size
        
        # First, resize dimensions if needed
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int((height * max_dimension) / width)
            else:
                new_height = max_dimension
                new_width = int((width * max_dimension) / height)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Resized to: {new_width}x{new_height}")
        
        # Convert to RGB if necessary
        if img.mode in ['RGBA', 'P']:
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[-1])
            img = background
        
        # Save to temporary file with quality optimization
        temp_path = _get_temp_image_path(original_path, "_comprehensive")
        qualities = [90, 80, 70, 60, 50, 40, 30, 25, 20]
        
        for quality in qualities:
            img.save(temp_path, format='JPEG', quality=quality, optimize=True)
            current_size = os.path.getsize(temp_path)
            
            if current_size <= target_size_bytes:
                current_size_mb = current_size / (1024 * 1024)
                logger.info(f"Comprehensive optimization complete: "
                           f"{img.size[0]}x{img.size[1]}, quality={quality}, {current_size_mb:.2f}MB")
                logger.info(f"Temporary optimized image saved to: {temp_path}")
                
                # Encode the optimized file
                with open(temp_path, "rb") as f:
                    encoded_string = base64.b64encode(f.read()).decode('utf-8')
                
                return encoded_string, temp_path
        
        # Last resort: further dimension reduction
        logger.warning("Standard optimization failed, trying aggressive dimension reduction")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)  # Clean up failed attempt
        
        return _compress_with_dimension_reduction_temp(img, original_path, target_size_mb, min_quality=15)
        
    except Exception as e:
        # Clean up temp file if it was created
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise
