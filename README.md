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
├── 📄 .gitignore # gitignore file for handling external files and directories
├── 📄 .neuro_symbolic_classifier_streamlit.py # Python file for running the Streamlit application
├── 📄 requirements.txt # Environment details necessary to run the experiments
├── 📄 README.md # Project documentation and instructions
```

## Live Demo
To run the Streamlit Demo simply click the link [here](https://neuro-symbolic-image-classifier-lix6pwwt9wnezutxgkkcks.streamlit.app). <\br>
Or if you would rather see the Gradio Demo on Google Colab, then click the link below. <\br>
To run the Gradio App Demo simply run this [notebook](https://colab.research.google.com/drive/1sIqVZL0FMISa4IXHp_psQ1B2hZnT7-hh?usp=sharing). <\br>