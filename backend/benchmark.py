"""
Comprehensive Benchmarking Suite for Sudoku Solver
Tests DLX solver performance, neural network inference, and end-to-end processing
"""

import time
import numpy as np
import os
from pathlib import Path
import sys

# Import our modules
from DLX import DLX
from neuralNetwork import NeuralNetwork
from mainSolver import solve_sudoku, sudoku_to_dlx

class SudokuBenchmark:
    def __init__(self):
        self.results = {}
        
    def get_memory_usage_mb(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def generate_test_puzzles(self, count=100):
        """Generate test Sudoku puzzles with varying difficulty"""
        puzzles = []
        
        # Easy puzzle (more clues)
        easy = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ]
        
        # Medium puzzle
        medium = [
            [0, 0, 0, 0, 0, 0, 6, 8, 0],
            [0, 0, 0, 0, 7, 3, 0, 0, 9],
            [3, 0, 9, 0, 0, 0, 0, 4, 5],
            [4, 9, 0, 0, 0, 0, 0, 0, 0],
            [8, 0, 3, 0, 5, 0, 9, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 3, 6],
            [9, 6, 0, 0, 0, 0, 3, 0, 8],
            [7, 0, 0, 6, 8, 0, 0, 0, 0],
            [0, 2, 8, 0, 0, 0, 0, 0, 0]
        ]
        
        # Hard puzzle (fewer clues)
        hard = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 0, 8, 5],
            [0, 0, 1, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 5, 0, 7, 0, 0, 0],
            [0, 0, 4, 0, 0, 0, 1, 0, 0],
            [0, 9, 0, 0, 0, 0, 0, 0, 0],
            [5, 0, 0, 0, 0, 0, 0, 7, 3],
            [0, 0, 2, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 4, 0, 0, 0, 9]
        ]
        
        # Distribute puzzles
        for i in range(count):
            if i % 3 == 0:
                puzzles.append(easy)
            elif i % 3 == 1:
                puzzles.append(medium)
            else:
                puzzles.append(hard)
        
        return puzzles
    
    def benchmark_dlx_solver(self, num_tests=100):
        """Benchmark the DLX solver performance"""
        print("\n" + "="*60)
        print("BENCHMARKING DLX SOLVER")
        print("="*60)
        
        puzzles = self.generate_test_puzzles(num_tests)
        
        times = []
        empty_cells = []
        
        for i, puzzle in enumerate(puzzles):
            # Count empty cells
            empty_count = sum(row.count(0) for row in puzzle)
            empty_cells.append(empty_count)
            
            # Time the solve
            start = time.perf_counter()
            solution = solve_sudoku(puzzle)
            end = time.perf_counter()
            
            solve_time = (end - start) * 1000  # Convert to milliseconds
            times.append(solve_time)
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{num_tests} puzzles solved...")
        
        # Calculate statistics
        avg_time = np.mean(times)
        median_time = np.median(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        avg_empty = np.mean(empty_cells)
        
        print(f"\n[RESULTS] DLX Solver Results ({num_tests} puzzles):")
        print(f"  Average solve time:  {avg_time:.2f} ms")
        print(f"  Median solve time:   {median_time:.2f} ms")
        print(f"  Min solve time:      {min_time:.2f} ms")
        print(f"  Max solve time:      {max_time:.2f} ms")
        print(f"  Std deviation:       {std_time:.2f} ms")
        print(f"  Avg empty cells:     {avg_empty:.1f}")
        
        self.results['dlx_avg_ms'] = avg_time
        self.results['dlx_median_ms'] = median_time
        self.results['dlx_max_ms'] = max_time
        
        return avg_time
    
    def benchmark_neural_network(self, num_tests=100):
        """Benchmark neural network inference speed"""
        print("\n" + "="*60)
        print("BENCHMARKING NEURAL NETWORK")
        print("="*60)
        
        # Initialize neural network
        print("  Loading neural network weights...")
        nn = NeuralNetwork()
        
        # Load weights
        weights_file = Path(__file__).parent / "weights" / "weights_and_biases_IMPROVEDv2.txt"
        if not weights_file.exists():
            print("  [WARNING] Weights file not found. Skipping NN benchmark.")
            return None
        
        # Load weights from file
        with open(weights_file, 'r') as f:
            lines = f.readlines()
            
        # Parse weights (simplified - adjust based on actual format)
        nn.Weight_Initialization()
        
        # Generate random test images (28x28)
        test_images = np.random.rand(num_tests, 28, 28)
        
        times = []
        
        print(f"  Running {num_tests} inference tests...")
        for i, img in enumerate(test_images):
            # Flatten image
            img_flat = img.flatten() / 255.0
            
            # Time the inference
            start = time.perf_counter()
            nn.Update_InputTargets(img, 0)  # Dummy label
            out_j = nn.Forward_Input_Hidden()
            nn.Forward_Hidden_Output(out_j)
            end = time.perf_counter()
            
            inference_time = (end - start) * 1000  # Convert to ms
            times.append(inference_time)
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{num_tests} inferences...")
        
        # Calculate statistics
        avg_time = np.mean(times)
        median_time = np.median(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"\n[RESULTS] Neural Network Results ({num_tests} inferences):")
        print(f"  Average inference:   {avg_time:.3f} ms")
        print(f"  Median inference:    {median_time:.3f} ms")
        print(f"  Min inference:       {min_time:.3f} ms")
        print(f"  Max inference:       {max_time:.3f} ms")
        print(f"  Total for 81 cells:  {avg_time * 81:.2f} ms")
        
        self.results['nn_avg_ms'] = avg_time
        self.results['nn_81_cells_ms'] = avg_time * 81
        
        return avg_time
    
    def benchmark_end_to_end(self):
        """Estimate end-to-end processing time"""
        print("\n" + "="*60)
        print("END-TO-END ESTIMATE")
        print("="*60)
        
        # Estimate based on components
        preprocessing_time = 50  # Estimated preprocessing time in ms
        segmentation_time = 30   # Estimated segmentation time in ms
        nn_time = self.results.get('nn_81_cells_ms', 100)
        solver_time = self.results.get('dlx_avg_ms', 20)
        
        total_time = preprocessing_time + segmentation_time + nn_time + solver_time
        
        print(f"  Preprocessing:       ~{preprocessing_time} ms")
        print(f"  Segmentation:        ~{segmentation_time} ms")
        print(f"  NN (81 cells):       ~{nn_time:.2f} ms")
        print(f"  DLX Solver:          ~{solver_time:.2f} ms")
        print(f"  ─────────────────────────────")
        print(f"  Total (estimated):   ~{total_time:.2f} ms ({total_time/1000:.2f}s)")
        
        self.results['end_to_end_ms'] = total_time
        
        return total_time
    
    def run_all_benchmarks(self):
        """Run all benchmarks and display summary"""
        print("\n" + "="*60)
        print("SUDOKU SOLVER COMPREHENSIVE BENCHMARK")
        print("="*60)
        
        start_time = time.time()
        
        # Run benchmarks
        self.benchmark_dlx_solver(num_tests=100)
        self.benchmark_neural_network(num_tests=100)  # Commented out - requires proper NN setup
        # self.benchmark_memory_usage()
        self.benchmark_end_to_end()
        
        end_time = time.time()
        
        # Final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"  DLX Average Solve:   {self.results.get('dlx_avg_ms', 0):.2f} ms")
        # print(f"  Memory Footprint:    {self.results.get('memory_mb', 0):.2f} MB")
        print(f"  End-to-End (est):    {self.results.get('end_to_end_ms', 0):.2f} ms")
        print(f"\n  Benchmark completed in {end_time - start_time:.2f} seconds")
        print("="*60)
        
        return self.results


if __name__ == "__main__":
    benchmark = SudokuBenchmark()
    results = benchmark.run_all_benchmarks()
    
    print("\n[DONE] Benchmark complete! Use these values for the 'How It Works' page.")
