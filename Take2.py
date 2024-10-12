import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.special import expit

def sigmoid(x):
    return expit(x)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / len(y_true)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def argmax(x):
    return np.argmax(x, axis=1).reshape(-1)

def flatten_image(image):
    return np.array(image).flatten() / 255.0  

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, reg_lambda=0.01, momentum=0.9):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        self.reg_lambda = reg_lambda
        self.momentum = momentum
        self.prev_weights_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.prev_weights_hidden_output = np.zeros_like(self.weights_hidden_output)

    def train(self, X, y, learning_rate, epochs):
        m = X.shape[0]
        for epoch in range(epochs):
            for i in range(m):
                # Forward pass
                hidden_input = np.dot(X[i:i+1], self.weights_input_hidden) + self.bias_hidden
                hidden_output = sigmoid(hidden_input)

                final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
                final_output = softmax(final_input)

                # Backpropagation
                error = final_output - y[i:i+1]
                d_hidden_output = error.dot(self.weights_hidden_output.T)
                d_hidden_input = d_hidden_output * sigmoid_derivative(hidden_output)

                # L1/L2 regularization
                if self.reg_lambda > 0:
                    l1_reg = self.reg_lambda * np.sign(self.weights_input_hidden)
                    l2_reg = self.reg_lambda * self.weights_input_hidden
                    d_hidden_input += l1_reg + l2_reg

                    l1_reg = self.reg_lambda * np.sign(self.weights_hidden_output)
                    l2_reg = self.reg_lambda * self.weights_hidden_output
                    d_hidden_output += l1_reg + l2_reg

                # Update weights and biases with momentum
                self.prev_weights_input_hidden *= self.momentum
                self.prev_weights_hidden_output *= self.momentum

                self.weights_input_hidden -= learning_rate * (X[i:i+1].T.dot(d_hidden_input) + self.prev_weights_input_hidden)
                self.weights_hidden_output -= learning_rate * (hidden_output.T.dot(error) + self.prev_weights_hidden_output)

                self.bias_hidden -= learning_rate * d_hidden_input
                self.bias_output -= learning_rate * error

            if epoch % 10 == 0:
                hidden_output = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
                final_output = softmax(np.dot(hidden_output, self.weights_hidden_output) + self.bias_output)
                ce_loss = -np.sum(y * np.log(final_output)) / m
                l2_regularization = 0.5 * self.reg_lambda * (np.sum(np.square(self.weights_input_hidden)) + np.sum(np.square(self.weights_hidden_output)))
                total_loss = ce_loss + l2_regularization
                print(f"Epoch {epoch}, Cross-Entropy Loss: {ce_loss}, Total Loss: {total_loss}")

    def predict(self, X):
        hidden_output = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        final_output = softmax(np.dot(hidden_output, self.weights_hidden_output) + self.bias_output)
        return np.argmax(final_output, axis=1)


def load_images_from_folder(folder):
    images = []
    labels = []

    class_folders = [class_folder for class_folder in os.listdir(folder) if os.path.isdir(os.path.join(folder, class_folder))]

    for class_folder in class_folders:
        class_path = os.path.join(folder, class_folder)
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            if img_path.endswith(('.jpg', '.jpeg', '.png')):
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize((64, 64))  # Resize image
                images.append(np.array(img))  
                labels.append(class_folder)

    return np.array(images), np.array(labels)

if __name__ == "__main__":
    train_folder_path = "/Users/abhudaysingh/Downloads/train"
    test_folder_path = "//Users/abhudaysingh/Downloads/test"
    train_images, train_labels = load_images_from_folder(train_folder_path)
    test_images, test_labels = load_images_from_folder(test_folder_path)

    label_encoder = LabelEncoder()
    encoded_train_labels = label_encoder.fit_transform(train_labels)
    encoded_test_labels = label_encoder.transform(test_labels)

    train_images = np.array([img.flatten() / 255.0 for img in train_images])
    test_images = np.array([img.flatten() / 255.0 for img in test_images])

    train_images, val_images, train_labels, val_labels = train_test_split(train_images, encoded_train_labels, test_size=0.2)

    # Normalize input data
    train_images = (train_images - np.mean(train_images)) / np.std(train_images)
    val_images = (val_images - np.mean(val_images)) / np.std(val_images)
    test_images = (test_images - np.mean(test_images)) / np.std(test_images)

    neural_net = NeuralNetwork(input_size=train_images.shape[1], hidden_size=64, output_size=len(np.unique(encoded_train_labels)), reg_lambda=0.001)
    neural_net.train(train_images, np.eye(len(np.unique(encoded_train_labels)))[train_labels], learning_rate=0.001, epochs=1000)

    val_predictions = neural_net.predict(val_images)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(f"Validation Accuracy: {val_accuracy}")

    test_predictions = neural_net.predict(test_images)
    test_accuracy = accuracy_score(encoded_test_labels, test_predictions)
    print(f"Test Accuracy: {test_accuracy}")