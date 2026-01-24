import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import your modules
from preProcessing import load_and_extract_sudoku
from Segmenting import extract_cells, clean_cell
from DLX import DLX
from neuralNetwork import NeuralNetwork

def image_to_array(image_path, weights_path):
    # --- 1. SETUP NEURAL NETWORK ---
    print("Loading Neural Network...")
    nn = NeuralNetwork()
    try:
        nn.Use_Trained_Weights(weights_path)
    except FileNotFoundError:
        print(f"Error: Weights file '{weights_path}' not found.")
        return None

    # --- 2. PREPROCESSING ---
    print(f"Processing {image_path}...")
    # We get the warped image (450x450) and validation status
    warped_img, grid_lines, is_valid = load_and_extract_sudoku(image_path)

    if not is_valid:
        print("Warning: Sudoku structure might be invalid, but attempting extraction anyway...")

    # --- 3. SEGMENTATION ---
    print("Extracting cells...")
    cells = extract_cells(warped_img, grid_lines) # Returns list of 81 cells (50x50)

    # Prepare the 9x9 grid
    sudoku_grid = np.zeros((9, 9), dtype=int)

    # --- 4. CLEANING & RECOGNITION ---
    print("Recognizing digits...")
    
    for i, cell in enumerate(cells):
        # Calculate row and col (0-8)
        row = i // 9
        col = i % 9

        # Clean the cell (get 28x28 centered digit)
        cleaned = clean_cell(cell)

        # OPTIMIZATION: If the cell is completely black (empty), skip the NN
        if np.sum(cleaned) == 0:
            sudoku_grid[row][col] = 0
            continue

        # Pass the 28x28 array to the Neural Network
        digit, confidence = nn.predict_from_array(cleaned)
        print(f"Cell ({row}, {col}): Detected digit {digit}, confidence {confidence:.4f}")
        sudoku_grid[row][col] = digit

    return sudoku_grid

def print_grid(grid):
    """
    Helper to print the grid nicely in the console.
    """
    print("\n--- DETECTED SUDOKU GRID ---")
    for r in range(9):
        if r % 3 == 0 and r != 0:
            print("-" * 21)
        for c in range(9):
            if c % 3 == 0 and c != 0:
                print("| ", end="")
            val = grid[r][c]
            print(f"{val if val != 0 else '.'} ", end="")
        print()
    print("----------------------------\n")
    

def sudoku_to_dlx(grid):
    """
    grid: 9x9 list of ints, 0 means empty
    returns: (dlx_instance, row_lookup)
    """

    # 324 primary columns
    columns = [(i, DLX.PRIMARY) for i in range(324)]
    solver = DLX(columns)

    solver.header = 324        # header index
    solver.partialsolution = []

    row_lookup = {}            # map row_id -> (r, c, n)

    def cell_col(r, c):
        return r * 9 + c

    def row_col(r, n):
        return 81 + r * 9 + n

    def col_col(c, n):
        return 162 + c * 9 + n

    def box_col(r, c, n):
        b = (r // 3) * 3 + (c // 3)
        return 243 + b * 9 + n

    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                nums = range(9)
            else:
                nums = [grid[r][c] - 1]

            for n in nums:
                cols = [
                    cell_col(r, c),
                    row_col(r, n),
                    col_col(c, n),
                    box_col(r, c, n)
                ]

                first = solver.appendRow(cols, (r, c, n))

                # walk the row and map ALL nodes to the same Sudoku meaning
                i = first
                while True:
                    row_lookup[i] = (r, c, n)
                    i = solver.R[i]
                    if i == first:
                        break

    return solver, row_lookup


def solve_sudoku(grid):
    solver, lookup = sudoku_to_dlx(grid)

    solution = solver.solve()
    if solution is None:
        return None

    result = [[0]*9 for _ in range(9)]
    result = [[0]*9 for _ in range(9)]
    for row_id in solution:
        r, c, n = lookup[row_id]
        result[r][c] = n + 1
    return result


    

if __name__ == "__main__":
    # CONFIGURATION
    img_path = "hope.png" 
    weights_file = "weights_and_biases_IMPROVEDv2.txt" # Ensure this file exists!

    # RUN THE SOLVER
    final_grid = image_to_array(img_path, weights_file)

    if final_grid is not None:
        print_grid(final_grid)
        
        # Optional: Print raw array for copying
        print("Raw Numpy Array:")
        print(repr(final_grid))
        
    solution = solve_sudoku(final_grid)

    for row in solution:
        print(row)