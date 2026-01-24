import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


class SudokuDataGenerator:
    """
    Generate synthetic Sudoku-style digits using all fonts in a folder
    """
    
    def __init__(self, fonts_folder="fonts"):
        """
        Initialize with path to fonts folder
        
        Args:
            fonts_folder: Path to folder containing .ttf or .otf font files
        """
        BASE_DIR = Path(__file__).resolve().parent
        if fonts_folder is None:
            self.fonts_folder = BASE_DIR / "Fonts"
        else:
            fonts_folder = Path(fonts_folder)
            self.fonts_folder = (
                fonts_folder if fonts_folder.is_absolute()
                else BASE_DIR / fonts_folder
            )

        self.DATA_DIR = BASE_DIR / "Data"
        self.DATA_DIR.mkdir(exist_ok=True)
        self.fonts_folder = fonts_folder
        self.font_paths = self._load_font_paths()
        
        if not self.font_paths:
            print(f"WARNING: No fonts found in '{fonts_folder}' folder!")
            print("Generator will use default system font instead.")
    
    def _load_font_paths(self):
        """
        Scan fonts folder and return list of all .ttf and .otf files
        """
        font_paths = []
        
        if not self.fonts_folder.exists():
            print(f"Fonts folder '{self.fonts_folder}' not found!")
            return font_paths
        
        # Supported font extensions
        font_extensions = ['.ttf', '.otf', '.TTF', '.OTF']
        
        # Walk through fonts folder (including subfolders)
        for path in self.fonts_folder.rglob("*"):
            if path.suffix.lower() in font_extensions:
                font_paths.append(str(path))
        
        print(f"Found {len(font_paths)} font files in '{self.fonts_folder}'")
        
        # Print first few fonts for verification
        if font_paths:
            print("Sample fonts loaded:")
            for font in font_paths[:5]:
                print(f"  - {Path(font).name}")
            if len(font_paths) > 5:
                print(f"  ... and {len(font_paths) - 5} more")
        
        return font_paths
    
    def generate_sudoku_digit(self, digit, size=28):
        """
        Generate synthetic Sudoku-style digit with realistic variations
        
        Args:
            digit: The digit to generate (0-9)
            size: Output image size (default 28x28)
        
        Returns:
            numpy array of shape (size, size) with values in [0, 1]
        """
        # Create larger canvas for better quality before downscaling
        canvas_size = size * 2
        img = Image.new('L', (canvas_size, canvas_size), color=255)  # White background
        draw = ImageDraw.Draw(img)
        
        # Select random font
        font = self._get_random_font(canvas_size)
        
        # Random position offset (simulate imperfect centering)
        offset_x = np.random.randint(-4, 5)
        offset_y = np.random.randint(-4, 5)
        
        # Draw the digit
        draw.text(
            (canvas_size // 2 + offset_x, canvas_size // 2 + offset_y),
            str(digit),
            fill=0,  # Black digit
            font=font,
            anchor='mm'
        )
        
        # Add grid lines (simulate Sudoku cell borders)
        self._add_grid_lines(draw, canvas_size)
        
        # Resize to target size
        img = img.resize((size, size), Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add realistic artifacts
        # img_array = self._add_artifacts(img_array)
        
        # Invert to match MNIST style (black background, white digit)
        img_array = 1.0 - img_array
        
        return img_array
    
    def _get_random_font(self, canvas_size):
        """
        Get a random font from the loaded fonts with random size
        """
        # Random font size (scaled to canvas)
        # font_size = np.random.randint(canvas_size // 3, canvas_size // 2)
        # Larger font sizes (fills more of the cell)
        font_size = np.random.randint(
            int(canvas_size * 0.8),
            int(canvas_size * 0.95)
        )
        
        if self.font_paths:
            # Try to load a random font from our collection
            font_path = np.random.choice(self.font_paths)
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception as e:
                # If loading fails, fall back to default
                print(f"Failed to load font {os.path.basename(font_path)}: {e}")
                return ImageFont.load_default()
        else:
            # Use default font if no custom fonts available
            return ImageFont.load_default()
    
    def _add_grid_lines(self, draw, canvas_size):
        """
        Add random grid lines to simulate Sudoku cell borders
        """
        if np.random.rand() > 0.3:  # 70% chance of grid lines
            line_thickness = np.random.randint(1, 4)
            line_color = np.random.randint(80, 180)  # Gray lines
            
            # Random horizontal line
            if np.random.rand() > 0.5:
                y_pos = np.random.choice([0, canvas_size - 1, np.random.randint(0, canvas_size)])
                draw.line([(0, y_pos), (canvas_size, y_pos)], fill=line_color, width=line_thickness)
            
            # Random vertical line
            if np.random.rand() > 0.5:
                x_pos = np.random.choice([0, canvas_size - 1, np.random.randint(0, canvas_size)])
                draw.line([(x_pos, 0), (x_pos, canvas_size)], fill=line_color, width=line_thickness)
    
    def _add_artifacts(self, img_array):
        """
        Add realistic artifacts: noise, compression, brightness variations
        """
        # Gaussian noise
        if np.random.rand() > 0.3:
            noise = np.random.normal(0, 0.02, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 1)
        
        # Random brightness/contrast
        brightness = np.random.uniform(0.7, 1.3)
        img_array = img_array * brightness
        
        # Random contrast
        contrast = np.random.uniform(0.8, 1.2)
        img_array = (img_array - 0.5) * contrast + 0.5
        
        # Blur (simulate scanning/compression)
        if np.random.rand() > 0.7:
            kernel_size = np.random.choice([3, 5])
            img_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
        
        # Clip to valid range
        img_array = np.clip(img_array, 0, 1)
        
        return img_array
    
    def generate_blank_cell(self, size=28):
        """
        Generate a blank Sudoku cell with noise and grid lines
        """
        # Start with mostly dark background
        img_array = np.random.uniform(0.0, 0.05, (size, size))
        
        # Add noise
        noise = np.random.normal(0, 0.08, (size, size))
        img_array += noise
        
        # Add faint grid lines
        if np.random.rand() > 0.4:
            thickness = np.random.randint(1, 2)
            intensity = np.random.uniform(0.1, 0.25)
            
            if np.random.rand() > 0.5:
                y = np.random.randint(0, size)
                img_array[y:y+thickness, :] += intensity
            else:
                x = np.random.randint(0, size)
                img_array[:, x:x+thickness] += intensity
        
        # Random smudges
        if np.random.rand() > 0.5:
            cx, cy = np.random.randint(0, size, 2)
            radius = np.random.randint(3, 7)
            strength = np.random.uniform(0.05, 0.15)
            
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                    if dist < radius:
                        img_array[i, j] += strength * (1 - dist / radius)
        
        # Clip to valid range
        img_array = np.clip(img_array, 0.0, 1.0)
        
        return img_array
    
    def generate_dataset(self, n_samples_per_digit=2000, include_blank=True):
        """
        Generate complete synthetic Sudoku dataset
        
        Args:
            n_samples_per_digit: Number of samples to generate per digit
            include_blank: Whether to include blank cells (class 10)
        
        Returns:
            images: numpy array of shape (n_total, 784)
            labels: numpy array of shape (n_total,)
        """
        images = []
        labels = []
        
        print(f"\nGenerating synthetic Sudoku dataset...")
        print(f"Samples per digit: {n_samples_per_digit}")
        
        # Generate digits 1-9 (Sudoku doesn't use 0)
        for digit in range(1, 10):
            print(f"Generating digit {digit}... ", end='', flush=True)
            for i in range(n_samples_per_digit):
                img = self.generate_sudoku_digit(digit)
                images.append(img.reshape(-1))  # Flatten to 784
                labels.append(digit)
            print("✓")
        
        # Generate blank cells
        if include_blank:
            print(f"Generating blank cells... ", end='', flush=True)
            for i in range(n_samples_per_digit):
                img = self.generate_blank_cell()
                images.append(img.reshape(-1))
                labels.append(10)  # Blank class
            print("✓")
        
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)
        
        print(f"\nDataset generated successfully!")
        print(f"Total images: {len(images)}")
        print(f"Image shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        
        return images, labels
    
    def generate_test_dataset(self, n_samples_per_digit=200, include_blank=True):
        """
        Generate test dataset (separate from training, different variations)
        
        Args:
            n_samples_per_digit: Number of test samples per digit
            include_blank: Whether to include blank cells
        
        Returns:
            test_images: numpy array of shape (n_total, 784)
            test_labels: numpy array of shape (n_total,)
        """
        images = []
        labels = []
        
        print(f"\nGenerating test dataset...")
        print(f"Test samples per digit: {n_samples_per_digit}")
        
        # Generate digits 1-9
        for digit in range(1, 10):
            print(f"Generating test digit {digit}... ", end='', flush=True)
            for i in range(n_samples_per_digit):
                img = self.generate_sudoku_digit(digit)
                images.append(img.reshape(-1))
                labels.append(digit)
            print("✓")
        
        # Generate blank cells
        if include_blank:
            print(f"Generating test blank cells... ", end='', flush=True)
            for i in range(n_samples_per_digit):
                img = self.generate_blank_cell()
                images.append(img.reshape(-1))
                labels.append(10)
            print("✓")
        
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)
        
        print(f"\nTest dataset generated successfully!")
        print(f"Total test images: {len(images)}")
        
        return images, labels

    def visualize_samples(self, n_samples=10):
        """
        Visualize random samples from each digit class
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        fig.suptitle('Sample Generated Sudoku Digits', fontsize=16)
        
        for idx, digit in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
            ax = axes[idx // 5, idx % 5]
            
            if digit == 10:
                img = self.generate_blank_cell()
                title = "Blank"
            else:
                img = self.generate_sudoku_digit(digit)
                title = str(digit)
            
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize generator with your fonts folder
    generator = SudokuDataGenerator(fonts_folder="fonts")
    
    # Generate dataset
    images, labels = generator.generate_dataset(
        n_samples_per_digit=2000,
        include_blank=True
    )
    
    # # Optionally visualize some samples
    generator.visualize_samples()
    
    # # Save to file for later use
    np.save(generator.DATA_DIR / 'sudoku_synthetic_images.npy', images)
    np.save(generator.DATA_DIR / 'sudoku_synthetic_labels.npy', labels)
    print("\nDataset saved to 'sudoku_synthetic_images.npy' and 'sudoku_synthetic_labels.npy'")

    # images, labels = generator.generate_test_dataset(n_samples_per_digit=200, include_blank=True)
    # np.save(generator.DATA_DIR / 'sudoku_test_images.npy',images)
    # np.save(generator.DATA_DIR / 'sudoku_test_labels.npy',labels)
    # print("\nDataset saved to 'sudoku_test_images.npy' and 'sudoku_test_labels.npy'")