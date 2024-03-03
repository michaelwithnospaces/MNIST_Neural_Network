import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def init_params(): # initial parameters
    # 784 columns represent the weigt of each pixel, 10 rows representing the hidden layer of neurons
    # subtract 0.5, .rand produces between [0, 1), shift to [-0.5, 0.5) to keep weights small initially
    # W1 dimensions: (10, 784) - 10 neurons, 784 input features (pixels)
    W1 = np.random.rand(10, 784) - 0.5 
    # 784 columns represent the weigt of each pixel, 1 row representing single base for the next layer of neurons
    # b1 dimensions: (10, 1) - bias for each of the 10 neurons
    b1 = np.random.rand(10, 1) - 0.5

    # Second layer
    # W2 dimensions: (10, 10) - mapping from 10 neurons in the first hidden layer to 10 output classes
    W2 = np.random.rand(10, 10) - 0.5 
    # b2 dimensions: (10, 1) - bias for each of the 10 output classes
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2

def ReLU(Z):
    # Z dimensions: Variable, depends on the input layer or hidden layer size
    return np.maximum(0, Z) # compares 2 arrays, 0 (scalar) and Z and returns maximum of the 2 in array the size of Z

def derivative_ReLU(Z):
    # Z dimensions: Same as input Z dimensions, depends on the layer size
    return Z > 0 # boolean True is treated as 1 and False as 0

def softmax(Z):
    # Z dimensions before softmax: (10, number of samples) - 10 output classes, each column is a sample
    #By subtracting the maximum value in Z from every element in Z, the largest value in the transformed Z becomes 0
    Z_exp = np.exp(Z - np.max(Z))

    # axis = 0 insures sum of columns keepdims keeps dimension
    """ 
    np.sum(a, axis=1, keepdims=True)
         array([[0],
                [1],
                [2],
                [1],
                [2]])
    np.sum(a, axis=1, keepdims=False)
         array([0, 1, 2, 1, 2]) 
    """
    return Z_exp / np.sum(Z_exp, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X): # X is input array
    # X dimensions: (784, number of samples) - 784 features (pixels), each column is a sample
    Z1 = W1.dot(X) + b1 # Z1 dimensions: (10, number of samples)
    A1 = ReLU(Z1) # A1 dimensions: (10, number of samples)
    Z2 = W2.dot(A1) + b2 # Z2 dimensions: (10, number of samples)
    A2 = softmax(Z2) # A2 dimensions: (10, number of samples)
    return Z1, A1, Z2, A2

def one_hot_encode(Y):
    # Y dimensions: (number of samples,) - vector of labels
    # creates as many rows as there are labels (Y.size) and as many columns as there are max numbers (+1 because indexing is from 0 - 9)
    one_hot_encoded_Y = np.zeros((Y.size, Y.max() + 1)) # (number of samples, 10) - 10 classes

    # indexing elements in a 2d array -> array[row, ith element]
    # np.arrange(Y.size) creates an array with the index of every element (as there are labels)
    # Y is the array of elements with labels
    # access' the row and sets the ith element (label) equal to 1
    one_hot_encoded_Y[np.arange(Y.size), Y] = 1
    one_hot_encoded_Y = one_hot_encoded_Y.T # Transposed: (10, number of samples)
    return one_hot_encoded_Y

def backward_prop(Z1, A1, Z2, A2, W2, X, Y): # Y represents our labels
    m = Y.size # Number of samples
    one_hot_encoded_y = one_hot_encode(Y) # (10, number of samples)
    dZ2 = A2 - one_hot_encoded_y # (10, number of samples)
    dW2 = 1 / m * dZ2.dot(A1.T) # (10, 10)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True) # (10, 1)
    dZ1 = W2.T.dot(dZ2) * derivative_ReLU(Z1) # (10, number of samples)
    dW1 = 1 / m * dZ1.dot(X.T) # (10, 784)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True) # (10, 1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    #  find the indices of the maximum values along axis 0 (column-wise). This effectively picks the class with the highest probability for each example.
    # A2 dimensions: (10, number of samples)
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    # predictions dimensions: (number of samples,), Y dimensions: (number of samples,)
    print(predictions, Y)
    # predictions == Y creates an array of True if correct and False if not
    # sums up all True (1) and divides by the total number of labels, to give overall correct
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        # check progress
        if (i % 50 == 0): # every 10th iteration
            print(f"Iteration: {i}")
            print(f"{get_accuracy(get_predictions(A2), Y) * 100.0:.4f}% Accuracy") # A2 is predictions (probabilities from forward prop)
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = Y_train[index]
    
    # Convert the current image to a 2D array and scale back to 0-255 pixel values
    current_image = current_image.reshape((28, 28)) * 255
    
    # Use matplotlib to display the image
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    
    # Set the title of the plot to include the prediction and the label
    plt.title(f"Prediction: {prediction} - Label: {label}")
    
    plt.show()


data = pd.read_csv('C:/Users/micha/Downloads/MNIST/data_sets/train.csv').to_numpy() # convert panda array to numpy array

np.random.shuffle(data) # shuffle data

data_dev = data[0:1000].T # collect first 1000 samples and transpose
Y_dev = data_dev[0].astype(int) # collect first row which is the label row and make sure type int for indexing
X_dev = data_dev[1:] # collect rest of the row which are features

data_train = data[1000:].T # collect rest of data for training model
Y_train = data_train[0].astype(int)
X_train = data_train[1:]

# Without normalization, the large input values for pixels could slow down learning or lead to suboptimal learning dynamics
X_train = X_train / 255.
X_dev = X_dev / 255.

print("Hi, I am Fred!")

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)

for i in range(100):
    test_prediction(i, W1, b1, W2, b2)
