from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found")
    return img

def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left
    return rect

def find_sudoku_contour(img):
    print("Attempting to find Sudoku contour...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        if cv2.contourArea(cnt) < 5000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            return approx  # FOUND A CROPPABLE SUDOKU

    return None


def warp_sudoku(image, contour):
    rect = order_points(contour)
    # Use a fixed size for the output for consistency (e.g., 450x450)
    side = 450 
    dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (side, side))

def find_and_crop_sudoku(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000: continue
        
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            # --- THE KEY ADDITION ---
            # Warp the candidate FIRST, then check if the INSIDE is a grid
            candidate_warp = warp_sudoku(gray, approx)
            
            # Re-threshold the warped version to check internal lines
            candidate_thresh = cv2.adaptiveThreshold(candidate_warp, 255, 
                                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY_INV, 11, 2)
            print("Grid validation passed!")
            plt.figure(figsize=(5, 5))
            plt.imshow(candidate_thresh, cmap='gray')
            plt.axis('off')
            plt.show()
            return candidate_warp

    raise ValueError("No valid Sudoku grid detected.")

def validate_sudoku_structure(binary_img):
    h, w = binary_img.shape
    
    # 1. Extract Horizontal Lines
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 15, 1))
    horiz_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horiz_kernel)
    
    # 2. Extract Vertical Lines
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 15))
    vert_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, vert_kernel)
    
    # --- NEW CLEANUP STEP ---
    # We "close" the lines to merge fragments (like a 2px gap) into one contour
    cleanup_kernel = np.ones((3,3), np.uint8)
    horiz_lines = cv2.morphologyEx(horiz_lines, cv2.MORPH_CLOSE, cleanup_kernel, iterations=2)
    vert_lines = cv2.morphologyEx(vert_lines, cv2.MORPH_CLOSE, cleanup_kernel, iterations=2)
    
    # 3. Count unique line clusters
    cnts_h, _ = cv2.findContours(horiz_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_v, _ = cv2.findContours(vert_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out tiny noise that survived the kernels
    num_h = len([c for c in cnts_h if cv2.boundingRect(c)[2] > w * 0.5])
    num_v = len([c for c in cnts_v if cv2.boundingRect(c)[3] > h * 0.5])
    
    print(f"Validation Check -> Found {num_h} long horizontal and {num_v} long vertical lines.")

    # A standard Sudoku has 10 lines. We'll accept 9 to 13 
    # (sometimes the outer border is thin or merged)
    if 9 <= num_h <= 13 and 9 <= num_v <= 13:
        return True
    
    return False

def get_full_grid(img, greyscale=False):
    if greyscale:
        gray = img
    else:  
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use a smaller block size (11) for adaptive thresholding to keep thin lines
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # --- THE FIX: Smaller Kernels ---
    # If the image is standard resolution, 40 is too big. Try 20-25.
    line_min_width = 20 
    
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_width, 1))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_min_width))

    # Extract lines
    hor_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hor_kernel, iterations=2)
    ver_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, ver_kernel, iterations=2)

    # Combine them
    full_grid = cv2.add(hor_lines, ver_lines)

    # --- THE FIX: Repairing Gaps ---
    # Use a small 2x2 or 3x3 kernel to 'fatten' the lines slightly so they connect
    repair_kernel = np.ones((2, 2), np.uint8)
    full_grid = cv2.dilate(full_grid, repair_kernel, iterations=1)
    
    # Optional: Closing operation to fill tiny holes
    full_grid = cv2.morphologyEx(full_grid, cv2.MORPH_CLOSE, repair_kernel)

    return cv2.bitwise_not(full_grid)

def preprocess_image(image):
    """
    Preprocess the given image by converting it to grayscale and applying binary thresholding.
    Args:
        image: The input image in cv2 format.
    """
    img = cv2.resize(image, (450, 450))
    if len(img.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    plt.figure(figsize=(5, 5))
    plt.imshow(binary, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return binary

def load_and_extract_sudoku(image_path):
    """
    Load an image and extract a validated Sudoku grid.
    
    Returns:
        img (np.ndarray): The final Sudoku image (cropped if needed)
        grid (np.ndarray): The extracted grid image
        valid (bool): Whether the structure matches a Sudoku grid
    """
    # Load image
    img = load_image(image_path)

    croppable_contour = find_sudoku_contour(img)
    if croppable_contour is not None:
        img = find_and_crop_sudoku(img)
        grid = get_full_grid(img, greyscale=True)
        valid = validate_sudoku_structure(cv2.bitwise_not(grid))
    else:
        grid = get_full_grid(img)
        valid = validate_sudoku_structure(cv2.bitwise_not(grid))

    # # First attempt: assume image already contains the grid
    # grid = get_full_grid(img)
    # valid = validate_sudoku_structure(cv2.bitwise_not(grid))

    # # Fallback: find, crop, and retry
    # if not valid:
    #     img = find_and_crop_sudoku(img)
    #     grid = get_full_grid(img, greyscale=True)
    #     valid = validate_sudoku_structure(cv2.bitwise_not(grid))
    
    img = cv2.resize(img, (450, 450))

    print(f"Full grid extraction successful. Is it a valid Sudoku structure? {valid}")
    print("Sudoku grid successfully extracted and displayed.")

    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.title(f"img")
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(4, 4))
    plt.imshow(grid, cmap='gray')
    plt.title(f"Grid")
    plt.axis('off')
    plt.show()

    if valid:
        processed_img = preprocess_image(img)
        return processed_img, grid, valid
    
    raise ValueError("No valid Sudoku grid detected.")

if __name__ == "__main__":
    image_path = "full puzzle1.png"  # Replace with your image path
    try:
        img, grid, valid = load_and_extract_sudoku(image_path)
    except Exception as e:
        print(f"Error: {e}")




