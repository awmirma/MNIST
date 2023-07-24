# MNIST Image Classification Project
 This project is an implementation of a machine learning model for classifying handwritten digits from the MNIST dataset using TensorFlow/Keras. The model is a simple feedforward neural network that achieves high accuracy in recognizing the digits.

## Getting Started
 To get started with the project, follow the steps below:

### Prerequisites
- Python 3.6 or higher
- TensorFlow 2.x
- NumPy
- Matplotlib (optional for visualization)
 You can install the required dependencies using the following command:

```bash
pip install tensorflow numpy matplotlib
```
### Dataset
 The MNIST dataset is already included in the TensorFlow/Keras library, so you don't need to download it separately.

### Training the Model
 To train the model, execute the train_model.py script:

```bash
python train_model.py
```
 The script will load the dataset, preprocess the data, build the model, train it, and save the trained model weights to model_weights.h5. You can modify the hyperparameters, such as the number of epochs and batch size, in the script.

### Evaluating the Model
 To evaluate the model on the test set and print the accuracy, execute the processes.py script:

```bash
python processes.py
```
 The script will load the saved model weights from model_weights.h5, evaluate the model on the test set, and print the test accuracy.

### Visualizing Predictions
 If you want to visualize the predictions for some test images, you can use the predict_images function in processes.py. By default, it prints the predictions for the first 10 test images. You can modify the num_images parameter to display predictions for a different number of images.

### Model Architecture
 The model used for this project is a simple feedforward neural network with three layers:

1. Input Layer: A fully connected (dense) layer with 128 units and ReLU activation function.

2. Hidden Layer: Another fully connected layer with 64 units and ReLU activation function.

3. Output Layer: The final layer with 10 units (corresponding to the 10 digits) and a softmax activation function.

### Results
 After training the model for 10 epochs, the accuracy on the test set is approximately 97%.

### Further Improvements
- Experiment with different model architectures, such as convolutional neural networks (CNNs), to potentially improve accuracy.

- Try hyperparameter tuning to optimize the model's performance.

- Visualize the model's predictions on a larger set of test images to gain insights into its strengths and weaknesses.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments
- The MNIST dataset is provided by TensorFlow/Keras and can be accessed through the tensorflow.keras.datasets.mnist module.

- Parts of the code structure and ideas are inspired by various online tutorials and resources.