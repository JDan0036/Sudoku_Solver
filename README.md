# Sudoku Solver

You can try out the live version of the project here: [Live Demo](https://sudoku-solver-website-mttb.onrender.com/)

## Project overview
This is a full stack web application that takes an image of an unsolved Sudoku puzzle, extracts the digits, and solves it instantly. It features a custom Optical Character Recognition engine built entirely from scratch and a highly optimized solving algorithm. In terms of performance, the neural network hits over 95% accuracy, the algorithmic solve time averages around 0.013 seconds, and the total end to end processing takes less than 0.35 seconds. 

## Problem and motivation
I built this because manually typing 81 numbers into a digital solver takes too much time. It defeats the purpose of quickly validating a puzzle. I felt that simply uploading an image would be a much better user experience. Beyond solving a practical problem, this project allowed me to dive deep into computer vision and figure out how neural networks actually function under the hood rather than just calling a method from a popular deep learning library. 

## Features
The main feature is the ability to parse an image of a Sudoku grid directly into a digital format. The backend handles perspective warping, cell segmentation, and digit recognition. 

If the image is blurry or a number is read incorrectly, the application still allows you to manually edit the grid before running the solver. 

To solve the puzzle, I completely avoided native backtracking. Instead, the solver translates the classic Sudoku rules into an exact cover problem and utilizes Donald Knuth's Dancing Links algorithm to find the solution. This is significantly faster and more efficient than a standard recursive approach.

## Tech stack
The backend is written in Python and exposed as a REST API using Flask. I used OpenCV for the heavier computer vision tasks like contour detection and perspective transformations. For the digit recognition, I built a dense neural network entirely from scratch using only Numpy. The frontend interface is built with React allowing users to easily upload images and view the solved grid. 

## Architecture overview
The flow of data is straightforward. A user uploads an image through the React frontend, which sends a base64 string to the Flask API. OpenCV takes over to find the largest square contour in the image, warps it to a perfect square, and slices it down into 81 individual segments. 

Each cell is cleaned up and evaluated. If a cell is not empty, it gets passed through our custom trained Numpy neural network to classify the digit. Once we have a complete 9x9 array, the Python backend translates the board state into a massive sparse matrix. The Dancing Links algorithm then traverses the matrix to satisfy all Sudoku constraints simultaneously. The completed grid is finally returned as a JSON response to the client.

## Getting started instructions
To run this project locally, make sure you have Python installed on your machine. Clone the repository and navigate into the backend folder. 

I recommend creating a virtual environment to keep your dependencies clean. You can do this by running `python -m venv venv` and activating it. 

After that, install the required packages by running `pip install -r requirements.txt`. 

Finally, you can start the local development server by running `python api.py`. The API will boot up and listen for requests on port 5000.

## Challenges and lessons learned
The hardest part of this project was dealing with the domain gap during the neural network training phase. I initially trained the model to recognize digits, but it struggled significantly when tested on actual printed Sudoku puzzles. The fonts found in puzzle books looked completely different from my initial training data. 

To solve this, I wrote a custom data generator. I generated synthetic images of standard computer fonts and applied random noise, smudges, and elastic deformations to simulate real world camera conditions. This completely turned the model around and brought the accuracy well over 95%. Building the network without PyTorch or TensorFlow was another major challenge, but it forced me to really understand gradient descent and matrix multiplication.

## Future improvements
While the solver is fast, I want to improve the initial preprocessing steps to handle extreme visual noise. Currently, severe shadows or bad lighting can obscure the grid lines and cause the perspective warp to fail. Improving the adaptive thresholding will make the initial image extraction step much more robust.

## License
This project is open source and available under the [MIT License](LICENSE).

## Contact Information
**Jobin Mathew Dan**  
LinkedIn: [https://www.linkedin.com/in/jobin-mathew-dan-7206412a0/]
Email: [jobindan2003@gmail.com]
