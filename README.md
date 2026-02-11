# Sudoku Solver with OCR and DLX Algorithm

A full-stack Sudoku solver that combines computer vision, custom neural networks, and Dancing Links (DLX) algorithm to solve puzzles from images. Built to address the tedious process of manual input by enabling users to simply upload a photo of their puzzle.

**Author:** Jobin Mathew Dan  
**Contact:** Jobindan2003@gmail.com  
**Live Demo:** [https://sudoku-solver-website-mttb.onrender.com](https://sudoku-solver-website-mttb.onrender.com)

---

## Problem and Motivation

Most Sudoku solvers require manual input of each cell, which is time-consuming and error-prone. I wanted to create something more practical. This project tackles that problem by accepting images as input, automatically extracting the grid, recognizing digits using a custom-built OCR system, and solving the puzzle using an efficient exact cover algorithm.

The real challenge wasn't just building a solver. It was building a robust OCR system that could handle real-world Sudoku images with varying fonts, lighting conditions, and grid styles. I trained a neural network from scratch and dealt with the domain gap between clean training data and messy real-world inputs.

---

## Features

- **Live web application**: Try it out at [sudoku-solver-website-mttb.onrender.com](https://sudoku-solver-website-mttb.onrender.com)
- **Image-based input**: Upload a photo of any Sudoku puzzle instead of typing it in manually
- **Custom OCR from scratch**: Built and trained a neural network to recognize digits 1-9 plus blank cells
- **Preprocessing pipeline**: Automatically detects, crops, and warps Sudoku grids from images using perspective transformation
- **DLX solver**: Implements Knuth's Dancing Links algorithm for efficient exact cover solving
- **Multiple solution detection**: Identifies if a puzzle has multiple valid solutions (up to 15)
- **REST API**: Flask backend with endpoints for image processing, digit verification, and puzzle solving
- **Benchmarking suite**: Comprehensive performance testing for both the neural network and DLX solver

---

## Tech Stack

**Backend:**
- Python 3.x
- Flask (REST API)
- OpenCV (image preprocessing and segmentation)
- NumPy (numerical operations and matrix manipulation)
- Pillow (image handling)

**Machine Learning:**
- Custom neural network implementation (no TensorFlow/PyTorch)
- MNIST dataset + synthetic Sudoku digit generation
- Data augmentation (rotation, scaling, elastic deformation, noise)

**Algorithms:**
- Dancing Links (DLX) for exact cover problem solving
- Adaptive thresholding and morphological operations for grid extraction
- Perspective transformation for image warping

**Deployment:**
- Gunicorn (production server)
- Docker support
- CORS-enabled for frontend integration

---

## Architecture Overview

The system follows a three-stage pipeline:

### 1. Image Preprocessing (`preProcessing.py`)
The preprocessing stage handles messy real-world images. It uses adaptive thresholding to handle varying lighting, finds the largest quadrilateral contour to detect the grid boundary, applies perspective transformation to warp the grid into a perfect square, and validates the structure by detecting horizontal and vertical grid lines.

### 2. Cell Segmentation (`Segmenting.py`)
Once we have a clean grid, the segmentation module divides it into 81 individual cells using morphological operations to remove grid lines. Each cell is then cleaned, centered, and resized to 28x28 pixels for the neural network.

### 3. Digit Recognition (`neuralNetwork.py`)
I built a custom neural network from scratch with 784 input neurons (28x28 pixels), 128 hidden neurons with ReLU activation, and 11 output neurons (digits 1-9 plus blank). The network uses Xavier initialization, dropout regularization, learning rate decay, and extensive data augmentation to handle the domain gap between MNIST and real Sudoku fonts.

### 4. Puzzle Solving (`DLX.py`, `mainSolver.py`)
The DLX algorithm treats Sudoku as an exact cover problem. It's significantly faster than naive backtracking, solving most puzzles in under 1 millisecond. The implementation can also detect multiple solutions, which is useful for validating puzzle uniqueness.

---

## Getting Started

### Prerequisites
```bash
Python 3.8+
pip
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/JDan0036/Sudoku_Solver.git
cd Sudoku_Solver
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Run the API server:
```bash
python api.py
```

The server will start on `http://localhost:5000`.

### API Endpoints

**Process Sudoku Image:**
```
POST /api/process-sudoku
Body: { "image": "base64_encoded_image" }
Returns: { "grid": [[...]], "confidences": [[...]] }
```

**Solve Sudoku:**
```
POST /api/solve-sudoku
Body: { "grid": [[...]] }
Returns: { "solutions": [[[...]]], "count": int }
```

**Health Check:**
```
GET /api/health
```

### Running Benchmarks

To see performance metrics:
```bash
python backend/benchmark.py
```

This will test the DLX solver on 100 puzzles and the neural network on 100 inferences, giving you average solve times and accuracy metrics.

---

## Challenges and Lessons Learned

The biggest challenge was the **domain gap** between training and real-world data. MNIST digits look very different from printed Sudoku numbers. Fonts vary widely, some are bold, some are thin, and real images have noise, shadows, and grid line artifacts.

I addressed this by generating synthetic Sudoku digits using multiple fonts, applying aggressive data augmentation during training (rotation, scaling, elastic deformation, noise), and training the network to recognize blank cells as a separate class. This improved real-world accuracy from around 85% to over 97%.

Another challenge was **grid extraction**. Not all Sudoku images are perfectly framed. Some have borders, some are skewed, and some have thick or thin grid lines. I built a robust preprocessing pipeline that uses contour detection and perspective transformation, but it still struggles with very low-contrast images or grids with unusual styling.

I also learned a lot about the trade-offs in algorithm design. The DLX implementation is elegant and fast, but it required careful thought about how to model Sudoku as an exact cover problem. The neural network taught me about the importance of regularization and data augmentation when you have limited training data.

---

## Future Improvements

There are a few areas I'd like to improve:

**Better preprocessing**: Add more sophisticated noise removal and contrast enhancement to handle poor-quality images. Maybe implement CLAHE (Contrast Limited Adaptive Histogram Equalization) or other advanced techniques.

**Model improvements**: Experiment with convolutional layers or a small CNN architecture instead of a fully connected network. This might improve accuracy on unusual fonts.

**Real-time feedback**: Add a confidence threshold system that flags low-confidence predictions and asks the user to verify them before solving.


---

## Performance

Based on benchmark results:

- **DLX Solver**: Average solve time of 0.013s (13ms) per puzzle
- **Neural Network**: 95%+ accuracy on real-world Sudoku digits
- **End-to-end processing**: < 0.35s from image upload to solved grid

---

## Acknowledgments

- Donald Knuth for the Dancing Links algorithm
- MNIST dataset for initial training data
- OpenCV community for excellent documentation
