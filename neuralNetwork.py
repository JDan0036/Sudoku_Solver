"""
This file contains code that will help train, test and run the neural network based on the numbers that are normally used in a sudoku puzzle.
This code is developed by Jobin Mathew Dan 
"""

import math
import json
import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import time
from PIL import Image, ImageFilter


class NeuralNetwork:
    """
    This is the actual neural network class that will be used to create, train and test the neural network
    """
    def __init__(self):
        """
        Initializes all the global variables required for the neural network
        """
        self.dataset_folder = os.path.join(Path.cwd(), "Datasets")

        self.epoch = None
        self.requiredAccuracy = None
        self.nr_correct = 0
        self.count = 0
        self.globalAccuracy = None
        self.learning_rate = 0.01

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

        # blank_images, blank_labels = self.generate_blank_dataset(6000)

        # self.training_images = np.vstack([train_images, blank_images])
        # self.training_labels = np.concatenate([train_labels, blank_labels])

        # self.training_images, self.training_labels = self.shuffle_dataset(
        #     self.training_images,
        #     self.training_labels
        # )
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

        # blank_images, blank_labels = self.generate_blank_dataset(1000)

        # self.testing_images = np.vstack([test_images, blank_images])
        # self.testing_labels = np.concatenate([test_labels, blank_labels])

        # self.testing_images, self.testing_labels = self.shuffle_dataset(
        #     self.testing_images,
        #     self.testing_labels
        # )

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
    def Forward_Input_Hidden(self):
        """
        The calculations for the net_j and out_j value at each hidden neuron. This is the forward propagation step
        from the input to hidden layer
        """
        net_j = self.bias_j + self.wji @ self.x
        out_j = np.maximum(0, net_j)  # ReLU function

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
    
    def TrainingLoop(self, epochs=1000, required_success=99.95):
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
        self.Read_Files_Training()
        self.Weight_Initialization()

        print("Starting Training Loop...")

        flag = True
        while flag:
            epoch_start = time.perf_counter()  # start timing
            for i in range(len(self.training_images)):
                # First we will update the input neurons and the target outputs based on the current image and label
                self.Update_InputTargets(self.training_images[i], self.training_labels[i])

                # Then we will carry out forward propagation from input to hidden layer
                out_j = self.Forward_Input_Hidden()

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
        with open('weights_and_biases_TEMP_noblank.txt', 'w') as file:
            file.write(json.dumps(details))
    
    def Use_Trained_Weights(self, file):
        """
        This function will set the weights and biases of the system to the pretrained model based on the filename
        that the weights and biases to be used are currently in
        """
        # First we open the file
        with open(file) as f:
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
        
    
if __name__ == "__main__":
    myNeuralNetwork = NeuralNetwork()

    # myNeuralNetwork.Read_Files_Training()
    # myNeuralNetwork.Read_Files_Testing()

    # TODO: THIS FUNCTIONS NEEDS TO BE CODED FIRST AND FULLY FINISHED 
    # BEFORE I CAN RUN IT

    # myNeuralNetwork.TrainingLoop()
    myNeuralNetwork.Use_Trained_Weights("weights_and_biases_TEMP_noblank.txt")
    # myNeuralNetwork.Run_Test_Data()

    # myNeuralNetwork.Use_Trained_Weights("weights_and_biases_TEMP.txt")
    # myNeuralNetwork.Read_Files_Training()
    result = myNeuralNetwork.Get_Expected_Output(
        "test5.png",
        show=True
    )

    # print("Prediction:", result)


