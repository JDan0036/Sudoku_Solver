"""
Flask API for Sudoku Solver
This API handles image upload, preprocessing, segmentation, and digit recognition
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from pathlib import Path
import traceback
from io import BytesIO
from PIL import Image

# Import our custom modules
from preProcessing import load_and_extract_sudoku
from Segmenting import extract_cells, clean_cell
from neuralNetwork import NeuralNetwork

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize Neural Network
nn = NeuralNetwork()
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_FILE = BASE_DIR / "weights" / "weights_and_biases_IMPROVEDv2.txt"

# Load pretrained weights
try:
    nn.Use_Trained_Weights(str(WEIGHTS_FILE))
    print("[SUCCESS] Neural network weights loaded successfully")
except Exception as e:
    print(f"[ERROR] Error loading weights: {e}")

def base64_to_image(base64_string):
    """
    Convert base64 string to OpenCV image
    """
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64 to bytes
    img_bytes = base64.b64decode(base64_string)
    
    # Convert to PIL Image
    pil_img = Image.open(BytesIO(img_bytes))
    
    # Convert to OpenCV format (BGR)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    return img

def image_to_base64(img):
    """
    Convert OpenCV image to base64 string
    """
    # Convert to PIL Image
    if len(img.shape) == 2:  # Grayscale
        pil_img = Image.fromarray(img)
    else:  # Color
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Convert to base64
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'message': 'Sudoku Solver API is running'
    })

@app.route('/api/process-sudoku', methods=['POST'])
def process_sudoku():
    """
    Main endpoint to process Sudoku image
    Expects: { "image": "base64_string" }
    Returns: { "grid": [[]], "confidences": [[]], "success": bool }
    """
    try:
        # Get image from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Convert base64 to image
        img = base64_to_image(data['image'])
        
        # Save temporarily for processing
        temp_path = BASE_DIR / "temp_sudoku.png"
        cv2.imwrite(str(temp_path), img)
        
        # Step 1: Preprocess and extract Sudoku grid
        print("[STEP 1] Preprocessing image...")
        processed_img, grid_img, is_valid = load_and_extract_sudoku(str(temp_path))
        
        if not is_valid:
            return jsonify({
                'success': False,
                'error': 'Could not detect a valid Sudoku grid in the image'
            }), 400
        
        # Step 2: Segment into cells
        print("[STEP 2] Segmenting into cells...")
        all_cells = extract_cells(processed_img, grid_img)
        
        # Step 3: Clean and recognize digits
        print("[STEP 3] Recognizing digits...")
        grid = []
        confidences = []
        
        for row in range(9):
            grid_row = []
            conf_row = []
            
            for col in range(9):
                cell_index = row * 9 + col
                cell = all_cells[cell_index]
                
                # Clean the cell
                cleaned_cell = clean_cell(cell)
                
                # Predict digit
                digit, confidence = nn.predict_from_array(cleaned_cell)
                
                grid_row.append(digit)
                conf_row.append(round(confidence, 4))
            
            grid.append(grid_row)
            confidences.append(conf_row)
        
        # Clean up temp file
        temp_path.unlink(missing_ok=True)
        
        print("[SUCCESS] Processing complete!")
        
        return jsonify({
            'success': True,
            'grid': grid,
            'confidences': confidences
        })
    
    except Exception as e:
        print(f"[ERROR] Error processing image: {e}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/verify-digit', methods=['POST'])
def verify_digit():
    """
    Endpoint to re-verify a single digit
    Expects: { "image": "base64_string", "row": int, "col": int }
    Returns: { "digit": int, "confidence": float }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Convert base64 to image
        img = base64_to_image(data['image'])
        
        # Ensure it's 28x28 grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28))
        
        # Predict
        digit, confidence = nn.predict_from_array(img)
        
        return jsonify({
            'success': True,
            'digit': digit,
            'confidence': round(confidence, 4)
        })
    
    except Exception as e:
        print(f"[ERROR] Error verifying digit: {e}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/solve-sudoku', methods=['POST'])
def solve_sudoku():
    """
    Solve a Sudoku puzzle using DLX algorithm
    Expects: { "grid": [[...]] }  (9x9 array with 0 for empty cells)
    Returns: { "solutions": [[[...]]], "count": int, "success": bool }
    """
    try:
        from DLX import DLX
        from mainSolver import sudoku_to_dlx
        
        data = request.get_json()
        
        if not data or 'grid' not in data:
            return jsonify({
                'success': False,
                'error': 'No grid provided'
            }), 400
        
        grid = data['grid']
        
        # Validate grid format
        if not isinstance(grid, list) or len(grid) != 9:
            return jsonify({
                'success': False,
                'error': 'Invalid grid format: must be 9x9 array'
            }), 400
        
        for row in grid:
            if not isinstance(row, list) or len(row) != 9:
                return jsonify({
                    'success': False,
                    'error': 'Invalid grid format: each row must have 9 cells'
                }), 400
        
        print("[STEP 1] Converting grid to DLX format...")
        solver, lookup = sudoku_to_dlx(grid)
        
        print("[STEP 2] Finding all solutions...")
        solutions = solver.solve_all()
        
        # Limit to top 10 solutions
        max_solutions = 10
        solutions = solutions[:max_solutions]
        
        print(f"[SUCCESS] Found {len(solutions)} solution(s)")
        
        # Convert solutions back to grid format
        all_grids = []
        for sol in solutions:
            grid_out = [[0]*9 for _ in range(9)]
            for row_id in sol:
                r, c, n = lookup[row_id]
                grid_out[r][c] = n + 1
            all_grids.append(grid_out)
        
        return jsonify({
            'success': True,
            'solutions': all_grids,
            'count': len(all_grids),
            'hasMultipleSolutions': len(solutions) > 1,
            'totalFound': len(solutions)
        })
    
    except Exception as e:
        print(f"[ERROR] Error solving Sudoku: {e}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Sudoku Solver API...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Weights file: {WEIGHTS_FILE}")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
