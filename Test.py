import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

def load_images_from_folder(folder):
    images = []
    labels = []

    class_folders = [class_folder for class_folder in os.listdir(folder) if os.path.isdir(os.path.join(folder, class_folder))]

    for class_folder in class_folders:
        class_path = os.path.join(folder, class_folder)
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            if img_path.endswith(('.jpg')):
                img = Image.open(img_path)
                images.append(np.array(img))  
                labels.append(class_folder)

    return np.array(images), np.array(labels)

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
        for epoch in range(epochs):
            hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_output = sigmoid(hidden_input)

            final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
            final_output = softmax(final_input)

            ce_loss = cross_entropy_loss(y, final_output)
            l2_regularization = 0.5 * self.reg_lambda * (np.sum(np.square(self.weights_input_hidden)) + np.sum(np.square(self.weights_hidden_output)))

            error = y - final_output
            output_delta = error * softmax_derivative(final_input)
            hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
            hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_output)

            self.prev_weights_input_hidden *= self.momentum
            self.prev_weights_hidden_output *= self.momentum

            self.weights_hidden_output -= hidden_output.T.dot(output_delta) * learning_rate + self.reg_lambda * self.weights_hidden_output
            self.bias_output -= np.sum(output_delta, axis=0, keepdims=True) * learning_rate
            self.weights_input_hidden -= X.T.dot(hidden_layer_delta) * learning_rate + self.reg_lambda * self.weights_input_hidden
            self.bias_hidden -= np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Cross-Entropy Loss: {ce_loss + l2_regularization}")

    def predict(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = sigmoid(hidden_input)

        final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = softmax(final_input)

        return argmax(final_output)

if __name__ == "__main__":
    train_folder_path = "/Users/abhudaysingh/Downloads/train"
    test_folder_path = "//Users/abhudaysingh/Downloads/test"

    train_images, train_labels = load_images_from_folder(train_folder_path)
    test_images, test_labels = load_images_from_folder(test_folder_path)

    label_encoder = LabelEncoder()
    encoded_train_labels = label_encoder.fit_transform(train_labels)
    encoded_test_labels = label_encoder.transform(test_labels)

    train_images = np.array([flatten_image(img) for img in train_images])
    test_images = np.array([flatten_image(img) for img in test_images])

    encoded_train_labels = encoded_train_labels.reshape(-1, 1)
    encoded_test_labels = encoded_test_labels.flatten()

    # Standardize input data
    scaler = StandardScaler()
    train_images_scaled = scaler.fit_transform(train_images)
    test_images_scaled = scaler.transform(test_images)

    train_images_scaled, val_images_scaled, train_labels, val_labels = train_test_split(train_images_scaled, encoded_train_labels, test_size=0.2)

    neural_net = NeuralNetwork(input_size=len(flatten_image(train_images_scaled[0])), hidden_size=16, output_size=len(np.unique(encoded_train_labels)))
    neural_net.train(train_images_scaled, train_labels, learning_rate=0.001, epochs=2000)

    val_predictions = neural_net.predict(val_images_scaled)
    val_predicted_labels = np.round(val_predictions)

    print("Unique values in val_labels:", np.unique(val_labels))
    print("Unique values in val_predicted_labels:", np.unique(val_predicted_labels))

    val_accuracy = accuracy_score(val_labels, val_predicted_labels)
    print(f"Validation Accuracy: {val_accuracy}")

    test_predictions = neural_net.predict(test_images_scaled)
    test_predicted_labels = np.round(test_predictions)

    test_accuracy = accuracy_score(encoded_test_labels, test_predicted_labels)
    print(f"Test Accuracy: {test_accuracy}")
