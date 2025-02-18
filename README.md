# neuro-symbolic-image-classifier
Project for a neuro-symbolic Image classifier utilizing a neural network and a ruleset induced from a decision tree to classfy images from the CIFAR-10 dataset in an explainable and efficient manner.

## 📂 Directory Structure

```
📂 project_root │
├── 📂 models # Pre-trained models for image classification
│ ├── 📄 feature_recognition_cnn.onnx # CNN model in ONNX format
│ ├── 📄 feature_recognition_cnn.pth # CNN model in PyTorch format
│ ├── 📄 neuro_symbolic_classifier.pkl # Pickled hybrid classifier
│
├── 📂 notebooks # Jupyter notebooks for training and inference
│ ├── 📄 Neurosymbolic_App.ipynb # Notebook for running the live demo app
│ ├── 📄 Neurosymbolic_Image_Classifier.ipynb # Notebook for training/testing the classifier
│
├── 📄 README.md # Project documentation and instructions
```

## Live Demo
To run the Gradio App Demo simply run this [notebook](https://colab.research.google.com/drive/1sIqVZL0FMISa4IXHp_psQ1B2hZnT7-hh?usp=sharing).