# Sudoku Solver Backend

This directory contains the entire backend stack for the Sudoku Solver project. It handles image processing (OpenCV), digit recognition (Custom Numpy Neural Network), and the algorithmic Sudoku solver (Dancing Links / DLX).

It can be run either as a standalone REST API (Flask) or executed directly via your terminal for local image testing.

## Prerequisites

Make sure you have Python 3.x installed. It is highly recommended to use a virtual environment.

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the API Server

You can start the standalone Flask server to accept HTTP requests by running:

```bash
python api.py
```

The REST API will boot up and listen for requests typically on `http://127.0.0.1:5000`. 

### Key Endpoints

- **`GET /api/health`**
  Simple health check to verify the API is running.

- **`POST /api/process-sudoku`**
  Expects a JSON payload with a base64 encoded image: 
  ```json
  { "image": "base64_string" }
  ```
  Returns the detected 9x9 grid array and the neural network's confidence levels.

- **`POST /api/solve-sudoku`**
  Expects a JSON payload containing the 9x9 grid mapping (use `0` for empty cells): 
  ```json
  { "grid": [[...]] }
  ```
  Returns the solved grid utilizing the DLX algorithmic solver.

## Running Locally via Terminal

If you just want to process a local image without starting up the API web server, you can use the `mainSolver.py` script.

1. Open `mainSolver.py` in your text editor and scroll to the bottom.
2. Update the `img_path` variable to point to the actual file path of your test image. For example: `img_path = BASE_DIR / "hope.png"`
3. Run the script from the terminal:
   ```bash
   python mainSolver.py
   ```
   
The script will run the full pipeline locally—preprocessing the image, detecting digits, and solving the puzzle. The raw array of the detected grid will be printed to your console along with the fully solved array.
