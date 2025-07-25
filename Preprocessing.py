import cv2
import numpy as np
import os


def normalize_image(image):
    """Normalize image pixel values to [0,1] range"""
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val + 1e-7)
    return normalized


def guided_filter(image, radius=5, eps=0.01):
    """
    Apply guided filter for edge-preserving smoothing
    :param radius: filter radius
    :param eps: regularization parameter
    """
    if len(image.shape) == 3:  # Color image
        guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Grayscale
        guide = image

    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    filtered = cv2.ximgproc.guidedFilter(
        guide=guide.astype(np.float32),
        src=image,
        radius=radius,
        eps=eps,
        dDepth=-1
    )

    if image.dtype == np.uint8:
        filtered = (filtered * 255).astype(np.uint8)

    return filtered


def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """Apply CLAHE to enhance contrast"""
    if len(image.shape) == 3:  # Color image
        # Convert to LAB color space and apply CLAHE to L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Convert L channel to 8-bit if needed
        if l.dtype != np.uint8:
            l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    else:  # Grayscale
        # Convert to 8-bit if needed
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        result = clahe.apply(image)
    return result


def gamma_correction(image, gamma=1.5, gain=1.0):
    """Apply gamma correction to adjust brightness"""
    # Build lookup table
    lookup_table = np.array([((i / 255.0) ** gamma) * 255 * gain
                             for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction
    if len(image.shape) == 3:  # Color image
        corrected = cv2.LUT(image, lookup_table)
    else:  # Grayscale
        corrected = cv2.LUT(image, lookup_table)

    return corrected


def preprocess_weld_image(image_path, output_path):
    """Complete preprocessing pipeline for weld flaw images"""
    try:
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Could not read image at {image_path}")
            return

        # Convert to grayscale (as weld flaw images are typically grayscale)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Normalization
        normalized = normalize_image(img)
        normalized = (normalized * 255).astype(np.uint8)

        # 2. Guided filtering
        filtered = guided_filter(normalized, radius=7, eps=0.03)

        # 3. CLAHE (now properly handles 8UC1 format)
        clahe_img = apply_clahe(filtered, clip_limit=2.0, grid_size=(8, 8))

        # 4. Gamma correction
        final_img = gamma_correction(clahe_img, gamma=1.2, gain=1.0)

        # Save result
        cv2.imwrite(output_path, final_img)
        return True

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False


def process_directory(input_dir, output_dir):
    """Process all images in input directory and save to output directory"""
    os.makedirs(output_dir, exist_ok=True)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process")
    success_count = 0

    for i, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_dir, filename)
        output_filename = f"processed_{filename}"
        output_path = os.path.join(output_dir, output_filename)

        if preprocess_weld_image(input_path, output_path):
            success_count += 1
            print(f"Processed {i}/{len(image_files)}: {filename}")
        else:
            print(f"Failed to process {i}/{len(image_files)}: {filename}")

    print(f"\nPreprocessing complete! Successfully processed {success_count}/{len(image_files)} images")


if __name__ == "__main__":
    input_directory = r'D:/weld_images'
    output_directory = r'D:/preprocessed_weld_images'

    process_directory(input_directory, output_directory)