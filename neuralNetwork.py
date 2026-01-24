"""
This file contains code that will help train, test and run the neural network based on the numbers that are normally used in a sudoku puzzle.
This code is developed by Jobin Mathew Dan 
"""

import math
import json
import os
from pathlib import Path
import numpy as np
from matplotlib import image, pyplot as plt
import cv2
import time
from PIL import Image, ImageFilter
from dataGenerator import SudokuDataGenerator


class NeuralNetwork:
    """
    This is the actual neural network class that will be used to create, train and test the neural network
    """
    def __init__(self):
        """
        Initializes all the global variables required for the neural network
        """
        BASE_DIR = Path(__file__).resolve().parent
        self.dataset_folder = BASE_DIR / "Datasets"
        self.fonts_path = BASE_DIR / "Fonts"
        self.WEIGHTS_DIR = BASE_DIR / "weights"
        self.WEIGHTS_DIR.mkdir(exist_ok=True)
        self.DATA_DIR = BASE_DIR / "data"

        self.epoch = None
        self.requiredAccuracy = None
        self.nr_correct = 0
        self.count = 0
        self.globalAccuracy = None
        self.learning_rate = 0.001
        self.initial_lr = 0.001

        self.input_neurons = 28 * 28  # 784 input neurons for 28x28 pixel images
        self.hidden_neurons = 128  # number of hidden neurons
        self.output_neurons = 11  # 11 output neurons for digits 0-9 + 1 blank
        self.targets = None  # target output vector

        self.x = None  # input layer
        self.wji = None  # weights from input to hidden layer
        self.wkj = None  # weights from hidden to output layer
        self.bias_j = None  # biases for hidden layer
        self.bias_k = None  # biases for output layer

        self.training_images = None
        self.testing_images = None
        self.training_labels = None
        self.testing_labels = None

    #done
    def Load_MNIST_Images(self, filename):
        """
        This function will load the MNIST images from the .gz file that are stored in the local directory
        """
        filepath = os.path.join(self.dataset_folder, filename)
        with open(filepath, 'rb') as f:
            # Read the magic number and dimensions
            magic_number = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            num_rows = int.from_bytes(f.read(4), 'big')
            num_cols = int.from_bytes(f.read(4), 'big')

            # Read the image data
            image_data = f.read(num_images * num_rows * num_cols)
            images = np.frombuffer(image_data, dtype=np.uint8)
            images = images.reshape((num_images, num_rows, num_cols))
        return images
    #done
    def Load_MNIST_Labels(self, filename):
        """
        This function will load the MNIST labels from the .gz file that are stored in the local directory
        """
        filepath = os.path.join(self.dataset_folder, filename)
        with open(filepath, 'rb') as f:
            # Read the magic number and number of labels
            magic_number = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')

            # Read the label data
            label_data = f.read(num_labels)
            labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels

    def generate_blank_cell(
        self,
        size=28,
        noise_level=0.08,
        grid_prob=0.6,
        smudge_prob=0.5
    ):
        """
        Generate a synthetic blank Sudoku cell with noise, faint grid lines,
        and smudges.
        """

        # 1. Dark background (not pure black)
        image = np.random.uniform(0.0, 0.05, (size, size))

        # 2. Gaussian noise
        noise = np.random.normal(0, noise_level, (size, size))
        image += noise

        # 3. Faint grid lines
        if np.random.rand() < grid_prob:
            thickness = np.random.randint(1, 2)
            intensity = np.random.uniform(0.1, 0.25)

            if np.random.rand() < 0.5:
                y = np.random.randint(0, size)
                image[y:y+thickness, :] += intensity
            else:
                x = np.random.randint(0, size)
                image[:, x:x+thickness] += intensity

        # 4. Smudges (blurred blobs)
        if np.random.rand() < smudge_prob:
            cx, cy = np.random.randint(0, size, 2)
            radius = np.random.randint(3, 7)
            strength = np.random.uniform(0.05, 0.15)

            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                    if dist < radius:
                        image[i, j] += strength * (1 - dist / radius)

        # 5. Clip to valid range
        image = np.clip(image, 0.0, 1.0)

        return image
    
    def generate_blank_dataset(self, n_samples):
        images = []
        labels = []

        for _ in range(n_samples):
            img = self.generate_blank_cell()
            images.append(img.reshape(-1))   # flatten to 784
            labels.append(10)                # blank class

        return np.array(images), np.array(labels)

    #done
    def preprocess_mnist(self, images):
        """
        Converts MNIST images to NN-ready format:
        - normalize to [0,1]
        - flatten to 1D vectors
        """
        images = images.astype(np.float32)
        images /= 255.0
        images = images.reshape(images.shape[0], -1)
        return images
    #done
    def shuffle_dataset(self, images, labels):
        indices = np.random.permutation(len(images))
        images = images[indices]
        labels = labels[indices]
        return images, labels
    #done
    def Read_Files_Training(self):
        """
        TODO: WRITE THE FOLLOWING DOCSTRING
        """
        train_images = self.Load_MNIST_Images('train-images-idx3-ubyte')
        train_labels = self.Load_MNIST_Labels('train-labels-idx1-ubyte')
        train_images = self.preprocess_mnist(train_images)

        self.training_images = train_images
        self.training_labels = train_labels

        blank_images, blank_labels = self.generate_blank_dataset(6000)

        self.training_images = np.vstack([train_images, blank_images])
        self.training_labels = np.concatenate([train_labels, blank_labels])

        self.training_images, self.training_labels = self.shuffle_dataset(
            self.training_images,
            self.training_labels
        )
    #done    
    def Read_Files_Testing(self):
        """
        TODO: WRITE THE FOLLOWING DOCSTRING
        """
        test_images = self.Load_MNIST_Images('t10k-images-idx3-ubyte')
        test_images = self.preprocess_mnist(test_images)
        test_labels = self.Load_MNIST_Labels('t10k-labels-idx1-ubyte')

        self.testing_images = test_images
        self.testing_labels = test_labels

        blank_images, blank_labels = self.generate_blank_dataset(1000)

        self.testing_images = np.vstack([test_images, blank_images])
        self.testing_labels = np.concatenate([test_labels, blank_labels])

        self.testing_images, self.testing_labels = self.shuffle_dataset(
            self.testing_images,
            self.testing_labels
        )

    def Train_with_synthetic_sudoku(self, n_samples=6000, epochs=100):
        """
        Train the neural network using synthetic Sudoku data
        """
      
        # Generate synthetic dataset
        generator = SudokuDataGenerator(fonts_folder=self.fonts_path)
        synthetic_images, synthetic_labels = generator.generate_dataset(
            n_samples_per_digit=n_samples,
            include_blank=True
        )
        
        # Combine with MNIST if desired
        # self.Read_Files_Training()  # Load MNIST first
        # self.training_images = np.vstack([self.training_images, synthetic_images])
        # self.training_labels = np.concatenate([self.training_labels, synthetic_labels])
        
        # Or train ONLY on synthetic data
        self.training_images = synthetic_images
        self.training_labels = synthetic_labels
        
        # Initialize weights and train
        self.Weight_Initialization()
        self.TrainingLoop(epochs=epochs, required_success=98.0)

    def Load_Synthetic_Training(self, images_path='sudoku_synthetic_images.npy', 
                             labels_path='sudoku_synthetic_labels.npy'):
        """
        Load pre-generated synthetic Sudoku training dataset
        
        Args:
            images_path: Path to .npy file containing training images
            labels_path: Path to .npy file containing training labels
        """
        print(f"Loading synthetic training data from {images_path}...")
        self.training_images = np.load(self.DATA_DIR / images_path)
        self.training_labels = np.load(self.DATA_DIR / labels_path)

        print(f"Training images loaded: {self.training_images.shape}")
        print(f"Training labels loaded: {self.training_labels.shape}")
        
        # Shuffle the dataset
        self.training_images, self.training_labels = self.shuffle_dataset(
            self.training_images,
            self.training_labels
        )
        print("Dataset shuffled successfully")

    def Load_Synthetic_Testing(self, images_path='sudoku_test_images.npy',
                                labels_path='sudoku_test_labels.npy'):
        """
        Load pre-generated synthetic Sudoku test dataset
        
        Args:
            images_path: Path to .npy file containing test images
            labels_path: Path to .npy file containing test labels
        """
        print(f"Loading synthetic test data from {images_path}...")
        self.testing_images = np.load(self.DATA_DIR / images_path)
        self.testing_labels = np.load(self.DATA_DIR / labels_path)

        print(f"Test images loaded: {self.testing_images.shape}")
        print(f"Test labels loaded: {self.testing_labels.shape}")

    def Test_Synthetic_Sudoku(self, show_errors=True, show_confusion_matrix=True):
        """
        Test the neural network on synthetic Sudoku dataset with detailed metrics
        
        Args:
            show_errors: If True, print misclassified examples
            show_confusion_matrix: If True, display confusion matrix
        
        Returns:
            accuracy: Overall accuracy percentage
        """
        print("\n" + "=" * 60)
        print("TESTING ON SYNTHETIC SUDOKU DATASET")
        print("=" * 60)
        
        correct = 0
        total = len(self.testing_images)
        
        # Track predictions for confusion matrix
        predictions = []
        true_labels = []
        errors = []
        
        # Per-digit accuracy tracking
        digit_correct = {i: 0 for i in range(1, 11)}  # 1-9 + 10 (blank)
        digit_total = {i: 0 for i in range(1, 11)}
        
        print(f"\nTesting {total} images...")
        
        for i in range(total):
            # Update input and targets
            self.Update_InputTargets(self.testing_images[i], self.testing_labels[i])
            
            # Forward propagation
            out_j = self.Forward_Input_Hidden()
            out_k = self.Forward_Hidden_Output(out_j)
            
            # Get prediction
            predicted_label = np.argmax(out_k)
            confidence = out_k[predicted_label, 0]
            true_label = self.testing_labels[i]
            
            predictions.append(predicted_label)
            true_labels.append(true_label)
            
            # Update counters
            digit_total[true_label] += 1
            
            if predicted_label == true_label:
                correct += 1
                digit_correct[true_label] += 1
            else:
                # Store error details
                errors.append({
                    'index': i,
                    'true': true_label,
                    'predicted': predicted_label,
                    'confidence': confidence
                })
            
            # Progress indicator
            if (i + 1) % 200 == 0:
                print(f"  Processed {i + 1}/{total} images...")
        
        # Calculate overall accuracy
        accuracy = (correct / total) * 100
        
        # Print results
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"Errors: {len(errors)}")
        
        # Per-digit accuracy
        print("\n" + "-" * 60)
        print("PER-DIGIT ACCURACY")
        print("-" * 60)
        print(f"{'Digit':<10} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
        print("-" * 60)
        
        for digit in range(1, 11):
            if digit_total[digit] > 0:
                digit_acc = (digit_correct[digit] / digit_total[digit]) * 100
                digit_name = "Blank" if digit == 10 else str(digit)
                print(f"{digit_name:<10} {digit_correct[digit]:<10} {digit_total[digit]:<10} {digit_acc:.2f}%")
        
        # Show sample errors
        if show_errors and errors:
            print("\n" + "-" * 60)
            print(f"SAMPLE ERRORS (showing first 10 of {len(errors)})")
            print("-" * 60)
            print(f"{'Index':<8} {'True':<8} {'Predicted':<12} {'Confidence':<12}")
            print("-" * 60)
            
            for error in errors[:10]:
                true_name = "Blank" if error['true'] == 10 else str(error['true'])
                pred_name = "Blank" if error['predicted'] == 10 else str(error['predicted'])
                print(f"{error['index']:<8} {true_name:<8} {pred_name:<12} {error['confidence']:.4f}")
        
        # Confusion matrix
        if show_confusion_matrix:
            self._plot_confusion_matrix(true_labels, predictions)
        
        return accuracy

    def _plot_confusion_matrix(self, true_labels, predictions):
        """
        Plot confusion matrix for test results
        
        Args:
            true_labels: List of true labels
            predictions: List of predicted labels
        """
        import matplotlib.pyplot as plt
        
        # Create confusion matrix
        classes = list(range(1, 11))  # 1-9 + 10 (blank)
        n_classes = len(classes)
        confusion = np.zeros((n_classes, n_classes), dtype=int)
        
        for true, pred in zip(true_labels, predictions):
            true_idx = true - 1
            pred_idx = pred - 1
            confusion[true_idx, pred_idx] += 1
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(confusion, cmap='Blues')
        
        # Labels
        class_names = [str(i) for i in range(1, 10)] + ['Blank']
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        # Rotate the tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                text = ax.text(j, i, confusion[i, j],
                            ha="center", va="center", 
                            color="white" if confusion[i, j] > confusion.max() / 2 else "black")
        
        ax.set_title("Confusion Matrix - Synthetic Sudoku Test")
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()

    def Train_On_Synthetic_Sudoku(self, epochs=100, required_success=97.5, 
                                load_from_file=True):
        """
        Complete training pipeline for synthetic Sudoku data
        
        Args:
            epochs: Maximum number of training epochs
            required_success: Target accuracy to stop training
            load_from_file: If True, load from .npy files, else generate new dataset
        """
        print("\n" + "=" * 60)
        print("TRAINING ON SYNTHETIC SUDOKU DATASET")
        print("=" * 60)
        
        if load_from_file:
            # Load pre-generated dataset
            self.Load_Synthetic_Training()
        else:
            # Generate new dataset
            generator = SudokuDataGenerator(fonts_folder=str(self.fonts_path))
            
            train_images, train_labels = generator.generate_dataset(
                n_samples_per_digit=1000,
                include_blank=True
            )
            
            self.training_images = train_images
            self.training_labels = train_labels
        
        # Initialize weights and train
        self.Weight_Initialization()
        self.TrainingLoop(epochs=epochs, required_success=required_success)
        
        # Save the trained model
        self.Saving_Weights_Bias()

    def image_resize(self):
        """
        This function will resize the input image of the character into the required size by first placing it on a
        square black canvas before resizing to prevent image distortion
        #TODO: CODE THE FOLLOWING FUNCTION
        """
        
    #done     
    def Weight_Initialization(self):
        """
        This will initialise the weights and biases as numpy arrays for efficiency using random inputs
        """
         # Xavier initialization for input -> hidden layer
        limit_ji = np.sqrt(6 / (self.input_neurons + self.hidden_neurons))
        self.wji = np.random.uniform(-limit_ji, limit_ji, (self.hidden_neurons, self.input_neurons))

        # Xavier initialization for hidden -> output layer
        limit_kj = np.sqrt(6 / (self.output_neurons + self.hidden_neurons))
        self.wkj = np.random.uniform(-limit_kj, limit_kj, (self.output_neurons, self.hidden_neurons))

        # Initialize biases to zero (you could also use small positive values if preferred)
        self.bias_j = np.zeros((self.hidden_neurons, 1))
        self.bias_k = np.zeros((self.output_neurons, 1))
    #done
    def Update_InputTargets(self, image, label):
        """
        This function will update the self.x which is the input neurons list based on the image that is passed in,
        and it will create the numpy array for the expected targets as well
        """
        # Input vector (column)
        self.x = image.reshape(-1, 1)

        # One-hot target vector
        self.targets = np.zeros((self.output_neurons, 1))
        self.targets[label, 0] = 1
    #done      
    def Forward_Input_Hidden(self, training=False, dropout_rate=0.3):
        """
        The calculations for the net_j and out_j value at each hidden neuron. This is the forward propagation step
        from the input to hidden layer
        """
        net_j = self.bias_j + self.wji @ self.x
        out_j = np.maximum(0, net_j)  # ReLU function

        if training:
            # Apply dropout during training
            mask = np.random.binomial(1, 1-dropout_rate, out_j.shape) / (1-dropout_rate)
            out_j *= mask

        return out_j
    #done     
    def Forward_Hidden_Output(self, out_j):
        """
        The calculations for the net_k and out_k at each output neuron. This is the forward propagation step
        from the hidden to output layer
        """
        net_k = self.bias_k + self.wkj @ out_j
        exp_x = np.exp(net_k - np.max(net_k, axis=0, keepdims=True))
        out_k = exp_x / np.sum(exp_x, axis=0, keepdims=True)  # Softmax function
        return out_k
    #done 
    def error_check(self, out_k):
        """
        This function will check to see how many of the predicted outputs are within 0.1 of the required targets. For
        every output which is within this error margin, the system will increment the nr_correct variable by 1. This
        variable will later be used to calculate the global accuracy
        """
        ERROR_MARGIN = 0.1

        self.nr_correct += np.sum((abs(out_k-self.targets) < ERROR_MARGIN) * 1)
    #done
    def Check_for_End(self):
        """
        This will return a boolean after checking whether the overall accuracy has reached the required level, or if
        the number of iterations run is more than the desired epochs
        """
        # Check whether the total error is less than the error set by the user or the number of iterations is reached
        # Return True or False

        # the global accuracy will be calculated based on the ratio of the total number of output neurons which were
        # correct to the total number of output neurons that were calculated (20 output neurons * 160 training images)
        self.globalAccuracy = round((self.nr_correct / (self.training_images.shape[0]*self.output_neurons)) * 100, 2)
        return self.globalAccuracy > self.requiredAccuracy or self.count >= self.epoch
    #done
    def Weight_Bias_Correction_Output(self, out_k):
        """
        This function will calculate the delta at the output layer, which will later be used to calculate the
        updated values of the wkj, and bias_k
        """
        delta_o = out_k - self.targets
        return delta_o
    #done
    def Weight_Bias_Correction_Hidden(self, out_j, delta_o):
        """
        This function will calculate the delta at the hidden layer, which will later be used to calculate the updated
        values of the wji, and bias_j
        """
        relu_derivative = (out_j > 0).astype(float)
        delta_h = (self.wkj.T @ delta_o) * relu_derivative
        return delta_h
    #done
    def Weight_Bias_Update(self, out_j, delta_o, delta_h):
        """
        This function will determine the updated values for the weights and biases required to make the network more
        accurate based on the latest iteration
        """
        # this section will use delta output previously calculated to calculate the updated values of the wkj and bias_k
        self.wkj += -self.learning_rate * delta_o @ np.transpose(out_j)
        self.bias_k += -self.learning_rate * delta_o

        # this section will use the delta hidden previously calculate to calculate the updated values of the wji and
        # bias_j
        self.wji += -self.learning_rate * delta_h @ np.transpose(self.x)
        self.bias_j += -self.learning_rate * delta_h
    
    def augment_image(self, image):
        """
        Apply random augmentations to make training more robust
        Uses only numpy and cv2 (already imported)
        """
        img = image.reshape(28, 28)
        
        # Random rotation (-15 to +15 degrees)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            center = (14, 14)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (28, 28), borderValue=0)
        
        # Random scaling (0.85 to 1.15)
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.85, 1.15)
            new_size = int(28 * scale)
            if new_size < 1:
                new_size = 1
            scaled = cv2.resize(img, (new_size, new_size))
            
            # Center crop/pad to 28x28
            if new_size > 28:
                start = (new_size - 28) // 2
                img = scaled[start:start+28, start:start+28]
            else:
                canvas = np.zeros((28, 28))
                start = (28 - new_size) // 2
                canvas[start:start+new_size, start:start+new_size] = scaled
                img = canvas
        
        # Random translation (shift by -3 to +3 pixels)
        if np.random.rand() > 0.5:
            shift_x = np.random.randint(-3, 4)
            shift_y = np.random.randint(-3, 4)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            img = cv2.warpAffine(img, M, (28, 28), borderValue=0)
        
        # Elastic deformation (using only NumPy and OpenCV)
        if np.random.rand() > 0.7:
            img = self.elastic_transform_cv2(img, alpha=8, sigma=3)
        
        # Random noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.05, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        # Random brightness adjustment
        if np.random.rand() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0, 1)
        
        return img.reshape(-1, 1)

    def elastic_transform_cv2(self, image, alpha, sigma):
        """
        Elastic deformation using only NumPy and OpenCV
        
        Args:
            image: 28x28 grayscale image (0-1 range)
            alpha: intensity of deformation (higher = more warping)
            sigma: smoothness of deformation (higher = smoother)
        
        Returns:
            Warped image
        """
        # Generate random displacement fields
        random_state = np.random.RandomState(None)
        shape = image.shape
        
        # Create random displacement fields
        dx = random_state.rand(*shape) * 2 - 1  # Range: -1 to 1
        dy = random_state.rand(*shape) * 2 - 1
        
        # Smooth the displacement fields using Gaussian blur (this replaces scipy's gaussian_filter)
        dx_smooth = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy_smooth = cv2.GaussianBlur(dy, (0, 0), sigma)
        
        # Scale by alpha
        dx_smooth *= alpha
        dy_smooth *= alpha
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        
        # Add displacement to coordinates
        map_x = (x + dx_smooth).astype(np.float32)
        map_y = (y + dy_smooth).astype(np.float32)
        
        # Apply remapping (this replaces scipy's map_coordinates)
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return warped

    def TrainingLoop(self, epochs=1000, required_success=97.5):
        """
        This function will run the entire training loop until either of the required conditions are met, by carrying
        out forward propagation, then backward propagation, then weights and bias updates during every epoch.

        In our case, we consider one epoch to be one iteration where every single image in the training dataset has
        run through the neural network once. In our case, since we have 160 training images, we consider 1 epoch to be
        where we have run through all 160 images once.
        """
        # First set the epoch and the required accuracy of the model
        self.epoch = epochs
        self.requiredAccuracy = required_success
        
        # Then we are going to read all the training files and initialize the weights and biases that are required
        # self.Read_Files_Training()
        self.Weight_Initialization()

        print("Starting Training Loop...")

        flag = True
        while flag:
            self.learning_rate = self.initial_lr * (0.95 ** (self.count // 10))
            epoch_start = time.perf_counter()  # start timing
            for i in range(len(self.training_images)):
                # First we will update the input neurons and the target outputs based on the current image and label
                # self.Update_InputTargets(self.training_images[i], self.training_labels[i])

                # With:
                if np.random.rand() > 0.5 and self.training_labels[i] != 10:  # Don't augment blanks
                    augmented = self.augment_image(self.training_images[i])
                    self.Update_InputTargets(augmented.flatten(), self.training_labels[i])
                else:
                    self.Update_InputTargets(self.training_images[i], self.training_labels[i])

                # Then we will carry out forward propagation from input to hidden layer
                out_j = self.Forward_Input_Hidden(training=True)

                # Then we will carry out forward propagation from hidden to output layer
                out_k = self.Forward_Hidden_Output(out_j)

                # Then we will check for error and update the nr_correct variable
                self.error_check(out_k)

                # Then we will calculate the delta at the output layer
                delta_o = self.Weight_Bias_Correction_Output(out_k)

                # Then we will calculate the delta at the hidden layer
                delta_h = self.Weight_Bias_Correction_Hidden(out_j, delta_o)

                # Finally, we will update the weights and biases based on the deltas calculated
                self.Weight_Bias_Update(out_j, delta_o, delta_h)

            epoch_end = time.perf_counter()  # end timing
            epoch_time = epoch_end - epoch_start
            # Increment the count variable by 1 after every epoch
            self.count += 1

            # Updated the flag to check for the end
            flag = not self.Check_for_End()

            # Print out the current epoch and accuracy
            print(f"Epoch: {self.count}")
            print(f"Accuracy: {self.globalAccuracy}")
            print(f"Epoch Time: {epoch_time:.4f} seconds")
            print("=================")

            # Reset the global accuracy and the nr_correct variables at the end of every epoch
            self.globalAccuracy = 0
            self.nr_correct = 0

        # Once the loop is complete, save the latest weights and biases
        self.Saving_Weights_Bias()
    
    def Saving_Weights_Bias(self):
        """
        This function will save all the latest weights and biases into a text file, so that it can be used later to
        run the neural network as a pretrained model
        """
        # Create a dictionary to save all the required data into text file
        # Later, the required data can be read from this dictionary
        details = {'wji': self.wji.tolist(),
                   'wkj': self.wkj.tolist(),
                   'bias_j': self.bias_j.tolist(),
                   'bias_k': self.bias_k.tolist()}

        # Write the data into a new text file to prevent from overriding any current files
        weights_path = self.WEIGHTS_DIR / "weights_and_biases_IMPROVEDv2.txt"
        with open(weights_path, 'w') as file:
            file.write(json.dumps(details))
    
    def Use_Trained_Weights(self, file):
        """
        This function will set the weights and biases of the system to the pretrained model based on the filename
        that the weights and biases to be used are currently in
        """
        # First we open the file
        if file is None:
            file_path = self.WEIGHTS_DIR / "weights_and_biases_IMPROVEDv2.txt"
        else:
            file_path = Path(file)
            if not file_path.is_absolute():
                file_path = self.WEIGHTS_DIR / file_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"Weights file not found: {file_path}")

        with open(file_path) as f:
            data = f.read()

        # Now we reconstruct the data that was found in the file back into a Python dictionary
        weights_and_biases = json.loads(data)
        self.wji = weights_and_biases['wji']
        self.wkj = weights_and_biases['wkj']
        self.bias_j = weights_and_biases['bias_j']
        self.bias_k = weights_and_biases['bias_k']
    
    def Run_Test_Data(self):
        """
        This function will run a loop to test the output of the neural network based on the 20% of manually cropped
        images that we did not use in the training. In our case, we have 20 test images to be used.
        """
        # first we read the testing files
        self.Read_Files_Testing()
        set = 0
        Correctness = 0

        # Now we run a loop to go through all the testing images
        for i in range(len(self.testing_images)):
            # First we will update the input neurons and the target outputs based on the current image and label
            self.Update_InputTargets(self.testing_images[i], self.testing_labels[i])

            # Then we will carry out forward propagation from input to hidden layer
            out_j = self.Forward_Input_Hidden()

            # Then we will carry out forward propagation from hidden to output layer
            out_k = self.Forward_Hidden_Output(out_j)
    
            # Now we will determine the predicted label using the max argument of the output neurons
            predicted_label = np.argmax(out_k)

            # Now we will print out the expected output and the actual output
            # print(f"Test Image {set+1}:")
            # print(f"Expected Output: {self.testing_labels[i]}")
            # print(f"Neural Network Output: {predicted_label}")
            # print("=================")
            set += 1
            Correctness += (predicted_label == self.testing_labels[i]) * 1
        accuracy = (Correctness / len(self.testing_images)) * 100
        print(f"Test Accuracy: {accuracy}%")

    def Get_Expected_Output(self,image_path, show=False):
        """
        This function will take in an input image, and run the pretrained neural network to see what is the expected
        output. Before using this, you MUST first upload the data of the pretrained model using the
        Use_Trained_Weights function. This function will return the output as a string
        """
        # 1. Load image
        img = Image.open(image_path).convert("L")  # grayscale
        img = img.filter(ImageFilter.SHARPEN)

        # 2. Resize while keeping aspect ratio
        max_side = max(img.size)
        square_img = Image.new("L", (max_side, max_side), color=0)  # black canvas
        square_img.paste(img, ((max_side - img.width) // 2, (max_side - img.height) // 2))
        img = square_img.resize((28, 28), Image.BILINEAR)

        # 3. Convert to numpy
        img = np.array(img, dtype=np.float32)

        # 4. Normalize to [0,1]
        img /= 255.0

        # 5. Apply threshold: below 0.5 -> black (0), above 0.5 -> white (1)
        # img = np.where(img > 0.5, 1.0, 0.0)

        # 6. Invert if background is white (MNIST-style)
        if img.mean() > 0.5:
            img = 1.0 - img

        # 7. Flatten
        img_flat = img.reshape(-1, 1)

        # 8. Forward pass
        self.x = img_flat
        out_j = self.Forward_Input_Hidden()
        out_k = self.Forward_Hidden_Output(out_j)

        # 9. Prediction
        predicted_label = np.argmax(out_k)

        # 10. Optional visualization
        if show:
            plt.imshow(img, cmap="gray")
            title = "Blank" if predicted_label == 10 else str(predicted_label)
            plt.title(f"Predicted: {title}")
            plt.axis("off")
            plt.show()

        return "Blank" if predicted_label == 10 else str(predicted_label)      

    # --- ADD THIS NEW METHOD TO NeuralNetwork Class ---
    def predict_from_array(self, img_array):
        """
        Takes a 28x28 numpy array (uint8, 0-255), normalizes it, 
        and returns the predicted digit.
        """
        # 1. Normalize to 0-1 range (Float)
        img = img_array.astype(np.float32) / 255.0

        # 2. Flatten to (784, 1) vector
        self.x = img.reshape(-1, 1)

        # 3. Forward Prop
        out_j = self.Forward_Input_Hidden()
        out_k = self.Forward_Hidden_Output(out_j)

        # 4. Get Prediction
        predicted_label = np.argmax(out_k)
        confidence = out_k[predicted_label, 0]

        # 5. Return 0 for "Blank" class (10) or the actual digit
        if predicted_label == 10:
            return 0, float(confidence)
        else:
            return int(predicted_label), float(confidence)

if __name__ == "__main__":
    myNeuralNetwork = NeuralNetwork()

    # myNeuralNetwork.Read_Files_Training()
    # myNeuralNetwork.Read_Files_Testing()

    # TODO: THIS FUNCTIONS NEEDS TO BE CODED FIRST AND FULLY FINISHED 
    # BEFORE I CAN RUN IT

    # myNeuralNetwork.TrainingLoop()
    myNeuralNetwork.Use_Trained_Weights("weights_and_biases_IMPROVEDv2.txt")
    # myNeuralNetwork.Run_Test_Data()

    # TRAIN WITH SYNTHETIC SUDOKU CELLS
    # myNeuralNetwork.Train_On_Synthetic_Sudoku(epochs=1000, required_success=98.0, load_from_file=True)
    myNeuralNetwork.Load_Synthetic_Testing()
    myNeuralNetwork.Test_Synthetic_Sudoku(show_errors=True, show_confusion_matrix=True)

    # myNeuralNetwork.Use_Trained_Weights("weights_and_biases_TEMP.txt")
    # # myNeuralNetwork.Read_Files_Training()
    # result = myNeuralNetwork.Get_Expected_Output(
    #     "test19.png",
    #     show=True
    # )

    # print("Prediction:", result)


