import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, reg_lambda=0.01):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        self.reg_lambda = reg_lambda

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            # Forward pass
            hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_output = 1 / (1 + np.exp(-hidden_input))

            final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
            final_output = np.exp(final_input - np.max(final_input, axis=1, keepdims=True))
            final_output /= np.sum(final_output, axis=1, keepdims=True)

            # Backpropagation
            error = final_output - y
            d_hidden_output = error.dot(self.weights_hidden_output.T)
            d_hidden_input = d_hidden_output * hidden_output * (1 - hidden_output)

            # Clip gradients during backpropagation
            d_hidden_input = np.clip(d_hidden_input, -1, 1)
            error = np.clip(error, -1, 1)

            # Update weights and biases
            self.weights_hidden_output -= hidden_output.T.dot(error) * learning_rate
            self.bias_output -= np.sum(error, axis=0, keepdims=True) * learning_rate
            self.weights_input_hidden -= X.T.dot(d_hidden_input) * learning_rate
            self.bias_hidden -= np.sum(d_hidden_input, axis=0, keepdims=True) * learning_rate

            if epoch % 10 == 0:
                ce_loss = -np.sum(y * np.log(final_output)) / len(y)
                print(f"Epoch {epoch}, Cross-Entropy Loss: {ce_loss}")

    def predict(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = 1 / (1 + np.exp(-hidden_input))

        final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = np.exp(final_input - np.max(final_input, axis=1, keepdims=True))
        final_output /= np.sum(final_output, axis=1, keepdims=True)

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

    neural_net = NeuralNetwork(input_size=train_images.shape[1], hidden_size=100, output_size=len(np.unique(encoded_train_labels)), reg_lambda=0.001)
    neural_net.train(train_images, np.eye(len(np.unique(encoded_train_labels)))[train_labels], learning_rate=0.001, epochs=1000)

    val_predictions = neural_net.predict(val_images)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(f"Validation Accuracy: {val_accuracy}")

    test_predictions = neural_net.predict(test_images)
    test_accuracy = accuracy_score(encoded_test_labels, test_predictions)
    print(f"Test Accuracy: {test_accuracy}")
