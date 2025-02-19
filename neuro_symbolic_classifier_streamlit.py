import streamlit as st
import torch
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelCNN(nn.Module):
    """
    A convolutional neural network for multi-label classification.

    Attributes:
    conv1, conv2, conv3 (nn.Conv2d): Convolutional layers.
    bn1, bn2, bn3 (nn.BatchNorm2d): Batch normalization layers.
    pool (nn.MaxPool2d): Max pooling layer.
    fc1, fc2, fc3 (nn.Linear): Fully connected layers.
    """
    def __init__(self, num_classes=10):
        """
        Initializes the MultiLabelCNN model.

        Parameters:
        num_classes (int, optional): Number of output classes. Defaults to 10.
        """
        super(MultiLabelCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        expected_size = self.get_expected_size()
        self.fc1 = nn.Linear(expected_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def get_expected_size(self):
        """
        Computes the output size after convolution and pooling layers.

        Returns:
        int: Flattened feature size before passing into fully connected layers.
        """
        device = next(self.parameters()).device
        random_input = torch.rand((1, 3, 32, 32), device=device)

        x = self.pool(F.relu(self.bn1(self.conv1(random_input))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        """
        Defines the forward pass of the CNN.

        Parameters:
        x (Tensor): Input tensor of shape (batch_size, 3, height, width).

        Returns:
        Tensor: Output logits for each class.
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
class NeuroSymbolicClassifier:
    """
    A hybrid classifier that combines neural network predictions with symbolic rule-based reasoning.

    The model first predicts a set of feature probabilities using the neural network, then converts them into
    binary values using a threshold. Based on this feature vector, a symbolic decision tree is used to predict
    the class. If a matching rule is found, it is returned; otherwise, a default message is returned.

    Parameters:
    neural_model (nn.Module): A trained neural network model for predicting feature probabilities.
    rules (list): A list of symbolic rules to be applied based on the predicted class.
    tree (sklearn.tree.DecisionTreeClassifier): A decision tree model used for classification based on feature vector.
    threshold (float): Threshold value for converting feature probabilities to binary values (default is 0.5).
    device (str): Device to run the model on, either "cpu" or "cuda" (default is "cpu").
    """
    def __init__(self, neural_model, rules, tree, threshold=0.5, device="cpu"):
        """
        Initializes the NeuroSymbolicClassifier.

        Parameters:
        neural_model (nn.Module): The trained neural network.
        rules (list): The set of rules to use with symbolic reasoning.
        tree (sklearn.tree.DecisionTreeClassifier): The decision tree for class prediction based on the binary feature vector.
        threshold (float): Threshold to determine the binary classification of each feature.
        device (str): The device on which the neural model is run (either "cpu" or "cuda").
        """
        neural_model.to(device)
        self.neural_model = neural_model
        self.rules = rules
        self.tree = tree
        self.threshold = threshold

    def _convert_to_binary(self, feature_probs):
        """
        Converts predicted feature probabilities to binary values based on a threshold.

        Parameters:
        feature_probs (list): The list of predicted feature probabilities from the neural network.

        Returns:
        tuple: A tuple of binary values (0 or 1) based on the threshold.
        """
        return tuple(int(val >= self.threshold) for val in feature_probs)

    def _find_matching_rule(self, predicted_class):
        """
        Searches for a matching symbolic rule corresponding to the predicted class.

        Parameters:
        predicted_class (str): The predicted class from the decision tree.

        Returns:
        str: The matching rule, or "NO MATCHING RULE" if no rule is found.
        """
        for rule in self.rules:
            if predicted_class in rule:
                return rule
        return "NO MATCHING RULE"

    def predict(self, image):
        """
        Makes a prediction using the neural model, decision tree, and symbolic rules.

        Parameters:
        image (ndarray or tensor): The input image for which a prediction is made.

        Returns:
        tuple: A tuple containing the predicted class and the applied rule (if any).
        """
        image = torch.tensor(image)
        image.to(device)
        self.neural_model.eval()
        with torch.no_grad():

            feature_probs = self.neural_model(image).squeeze().tolist()
        feature_vector = np.array(self._convert_to_binary(feature_probs), dtype=np.int8)
        predicted_class = str(self.tree.predict([feature_vector])[0])
        rule = self._find_matching_rule(predicted_class)
        return predicted_class, rule

# Load the trained classifier
@st.cache_resource()
def load_model():
    with open("./models/neuro_symbolic_classifier.pkl", "rb") as f:
        return pickle.load(f)

hybrid_classifier = load_model()

device = "cuda" if torch.cuda.is_available() else "cpu"

def classify_image(selected_class):
    """
    Selects a random image from the specified class, processes it,
    classifies it using a hybrid classifier, and returns the image along
    with the predicted class and applied rule.
    """
    class_images = imgs_per_class[selected_class]
    image = random.choice(class_images)
    image_input = np.expand_dims(image, axis=0)
    predicted_class, applied_rule = hybrid_classifier.predict(image_input)
    
    return image, predicted_class, applied_rule

# Load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@st.cache_resource()
def load_dataset():
    dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return dataset

testset = load_dataset()

@st.cache_resource()
def get_preprocessed_images_per_class():
    classes = testset.classes
    imgs_per_class = {cls: [] for cls in classes}

    for img, img_class in testset:
        imgs_per_class[classes[img_class]].append(img.numpy().copy())
    return classes, imgs_per_class

classes, imgs_per_class = get_preprocessed_images_per_class()

# Streamlit UI
st.title("🧠 Neuro-Symbolic Image Classifier")
st.markdown("""
This application demonstrates a **hybrid neuro-symbolic classifier** trained on the CIFAR-10 dataset.
Select a class, and the app will randomly choose an image, classify it, and display the result.
""")

selected_class = st.selectbox("Select a class", classes)

if st.button("Classify Image"):
    with st.spinner("Processing..."):
        image, predicted_class, applied_rule = classify_image(selected_class)
        
        # Convert image for display
        img_display = image.transpose((1, 2, 0))
        img_display = img_display * 0.5 + 0.5  # Unnormalize
        
        fig, ax = plt.subplots()
        ax.imshow(img_display)
        ax.axis("off")
        ax.set_title(f"Predicted: {predicted_class}\nRule: {applied_rule}", fontsize=10)
        
        st.pyplot(fig)
