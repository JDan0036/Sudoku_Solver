import cv2
import numpy as np
from preProcessing import load_and_extract_sudoku
import matplotlib.pyplot as plt

def extract_cells(warped_img, grid_only):
    """
    Splits the 450x450 warped image into 81 individual 50x50 cells.
    """
    removed_grid_img = remove_grid_from_warped(warped_img, grid_only)
    cells = []
    # Ensure the image is exactly 450x450 for clean 50px slices
    side = removed_grid_img.shape[0] // 9
    
    for r in range(9):
        for c in range(9):
            start_y = r * side
            end_y = (r + 1) * side
            start_x = c * side
            end_x = (c + 1) * side
            
            cell = warped_img[start_y:end_y, start_x:end_x]
            cells.append(cell)
            
    return cells

def remove_grid_from_warped(warped_img, grid_img):
    """
    Removes the Sudoku grid from the warped image using the extracted grid.
    
    Args:
        warped_img (np.ndarray): Warped Sudoku image (grayscale or BGR)
        grid_img (np.ndarray): Extracted grid image (from get_full_grid)
        
    Returns:
        np.ndarray: Image with grid removed
    """

    # Ensure grayscale
    if len(warped_img.shape) == 3:
        warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    else:
        warped_gray = warped_img.copy()

    # Ensure grid is binary and inverted (grid lines = white)
    if len(grid_img.shape) == 3:
        grid_gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
    else:
        grid_gray = grid_img.copy()

    # ✅ CRITICAL FIX: Match sizes
    if grid_gray.shape != warped_gray.shape:
        grid_gray = cv2.resize(
            grid_gray,
            (warped_gray.shape[1], warped_gray.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    _, grid_mask = cv2.threshold(grid_gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Slightly thicken grid mask to fully remove lines
    kernel = np.ones((3, 3), np.uint8)
    grid_mask = cv2.dilate(grid_mask, kernel, iterations=1)

    # Remove grid using inpainting (BEST result)
    removed = cv2.inpaint(
        warped_gray,
        grid_mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )

    # plt.figure(figsize=(4, 4))
    # plt.imshow(removed, cmap='gray')
    # plt.title(f"Removed Grid")
    # plt.axis('off')
    # plt.show()

    return removed

def clean_cell(cell_img):
    """
    Cleans noise by keeping only the largest connected component 
    THAT IS ALSO LOCATED IN THE CENTER.
    """
    h, w = cell_img.shape
    
    # 1. SAFETY CROP: Remove just 4-5 pixels to sever connections between 
    # the digit and any grid lines.
    margin = 4 
    cleaned_cell = cell_img[margin:h-margin, margin:w-margin]
    
    # Re-fetch new dimensions after crop
    h_c, w_c = cleaned_cell.shape
    
    # 2. DILATE: Swell pixels to connect broken segments of a digit
    kernel = np.ones((2,2), np.uint8)
    cleaned_cell = cv2.dilate(cleaned_cell, kernel, iterations=1)

    # 3. COMPONENT ANALYSIS
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cleaned_cell, connectivity=8)
    
    # Define the "Center Zone"
    # A blob is valid only if its center is within this distance from the image center
    img_center_x, img_center_y = w_c // 2, h_c // 2
    max_distance_from_center = w_c * 0.3  # Allow deviation up to 30% from center
    
    min_pixel_area = 50 # Ignore small specks of noise
    
    valid_candidates = []

    # Loop through all components (skip 0 which is background)
    for i in range(1, nb_components):
        area = stats[i, cv2.CC_STAT_AREA]
        centerX, centerY = centroids[i]
        
        # Calculate distance of this blob from the true center
        dist = np.sqrt((centerX - img_center_x)**2 + (centerY - img_center_y)**2)

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w_b = stats[i, cv2.CC_STAT_WIDTH]
        h_b = stats[i, cv2.CC_STAT_HEIGHT]
        
        # ---- Bounding-box center overlap check ----
        center_margin = int(w_c * 0.15)  # 15% of cell width

        center_x1 = img_center_x - center_margin
        center_y1 = img_center_y - center_margin
        center_x2 = img_center_x + center_margin
        center_y2 = img_center_y + center_margin

        # Bounding box extents
        box_x1, box_y1 = x, y
        box_x2, box_y2 = x + w_b, y + h_b

        # Check overlap
        if not (
            box_x2 > center_x1 and
            box_x1 < center_x2 and
            box_y2 > center_y1 and
            box_y1 < center_y2
        ):
            continue

        # ---- NEW: Aspect ratio sanity check ----
        aspect_ratio = w_b / h_b
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            continue

        # LOGIC: 
        # 1. Must be big enough to be a digit
        # 2. Must be close enough to center (ignoring border noise)
        if area > min_pixel_area and dist < max_distance_from_center:
            valid_candidates.append((area, i)) # Store (size, label_index)

    # If no valid blobs found (cell is empty or only has borders), return blank
    if not valid_candidates:
        return np.zeros((28, 28), dtype=np.uint8)
    
    # 4. SELECT THE WINNER: Sort by area (largest first)
    valid_candidates.sort(key=lambda x: x[0], reverse=True)
    best_label = valid_candidates[0][1] # Get index of largest valid blob

    # Create mask for the winner
    digit_only = np.zeros(output.shape, dtype=np.uint8)
    digit_only[output == best_label] = 255

    # ---- Center-density check ----
    h_dig, w_dig = digit_only.shape

    center_size = int(w_dig * 0.2)  # 20% center window
    cx, cy = w_dig // 2, h_dig // 2

    center_roi = digit_only[
        cy - center_size//2 : cy + center_size//2,
        cx - center_size//2 : cx + center_size//2
    ]

    white_ratio = np.count_nonzero(center_roi) / center_roi.size

    if white_ratio < 0.02:  # 2% ink threshold
        return np.zeros((28, 28), dtype=np.uint8)

    # --- CENTERING & RESIZING (Standard logic) ---
    coords = cv2.findNonZero(digit_only)
    x, y, w_d, h_d = cv2.boundingRect(coords)
    digit_crop = digit_only[y:y+h_d, x:x+w_d]

    aspect_ratio = w_d / h_d
    if w_d > h_d:
        new_w = 20
        new_h = int(20 / aspect_ratio)
    else:
        new_h = 20
        new_w = int(20 * aspect_ratio)
    
    new_w, new_h = max(new_w, 1), max(new_h, 1)
    digit_rescaled = cv2.resize(digit_crop, (new_w, new_h))
    
    canvas = np.zeros((28, 28), dtype=np.uint8)
    off_x = (28 - new_w) // 2
    off_y = (28 - new_h) // 2
    canvas[off_y:off_y+new_h, off_x:off_x+new_w] = digit_rescaled
    
    return canvas


def show_cells_grid(cells, title="All Cells", cols=9):
    rows = int(np.ceil(len(cells) / cols))
    plt.figure(figsize=(cols, rows))

    for i, cell in enumerate(cells):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cell, cmap='gray')
        plt.axis('off')
        plt.title(i, fontsize=6)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    image_path = "hope.png"
    print(f"Starting extraction for {image_path}...")

    try:
        # Step 1: Use the robust loading function
        # This function handles the logic of finding, cropping, and validating the grid
        img_warped, grid_only, is_valid = load_and_extract_sudoku(image_path)

        if not is_valid:
            print("Warning: The extracted structure did not pass full Sudoku validation.")


        # Step 2: Split the 450x450 warped image into 81 cells
        all_cells = extract_cells(img_warped, grid_only)
        print(f"Successfully extracted {len(all_cells)} cells.")

        # Step 3: Visualize a sample cell (e.g., the 5th cell)
        # We clean it to see what the digit recognizer would actually see
        # sample_index = 41
        # sample_cell = clean_cell(all_cells[sample_index])
        # # sample_cell = all_cells[sample_index]
        
        # plt.figure(figsize=(4, 4))
        # plt.imshow(sample_cell, cmap='gray')
        # plt.title(f"Cleaned Cell (Index {sample_index})")
        # plt.axis('off')
        # plt.show()

        cleaned_cells = [clean_cell(c) for c in all_cells]
        show_cells_grid(cleaned_cells, title="Cleaned Sudoku Cells (0–80)")
        

    except Exception as e:
        print(f"An error occurred during segmentation: {e}")