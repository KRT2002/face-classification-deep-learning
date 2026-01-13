# Face Recognition Model - Gender Classification

A deep learning project implementing a binary face recognition system to classify faces by gender using transfer learning with VGG16 architecture.

## 📋 Project Overview

This project demonstrates the implementation of a CNN-based face recognition model that achieves high accuracy in gender classification. The model leverages transfer learning from VGG16 pre-trained on ImageNet and includes advanced data augmentation, performance monitoring, and comprehensive evaluation metrics.

## 🎯 Key Features

- **Transfer Learning**: Utilizes VGG16 architecture pre-trained on ImageNet
- **Enhanced Data Augmentation**: Random flips, rotations, zooms, and contrast adjustments
- **Model Checkpointing**: Automatic saving of best-performing model based on validation loss
- **Comprehensive Evaluation**: Classification reports, confusion matrix, and visual analysis
- **Performance Visualization**: Training/validation metrics tracking and prediction display

## 🛠️ Tech Stack

- **Framework**: TensorFlow/Keras
- **Base Model**: VGG16 (Pre-trained on ImageNet)
- **Libraries**: 
  - TensorFlow Datasets
  - NumPy
  - Matplotlib
  - Scikit-learn
  - Pickle (for model persistence)
- **Image Processing**: TensorFlow image preprocessing
- **Data Augmentation**: Keras data augmentation layers

## 📊 Model Architecture

- **Base**: VGG16 (frozen layers)
- **Custom Layers**:
  - Global Average Pooling
  - Dense layers (256, 128 units) with ReLU activation
  - Dropout (0.5) for regularization
  - Output layer (2 units) with Softmax activation

## 🚀 Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/KRT2002/face-classification-deep-learning.git
cd face-classification-deep-learning
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the notebook**
Open and execute the Jupyter notebook to train and evaluate the model.

## 📈 Results

The model demonstrates strong performance across both classes:

- **Overall Accuracy**: ~95%+
- **Precision & Recall**: High scores for both genders
- **Model Generalization**: Validation metrics closely track training metrics, indicating good generalization without overfitting

For detailed results, metrics, and visualizations, refer to [DOCUMENTATION.md](DOCUMENTATION.md).

## 📁 Project Structure

```
├── Face Recognition model.docx    # Original documentation
├── DOCUMENTATION.md               # Detailed implementation guide
├── notebook.ipynb                 # Implementation notebook
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 💡 Implementation Highlights

1. **Data Preprocessing**: Images resized to 224x224, normalized and batched
2. **Class Balance Handling**: Dataset analysis shows balanced distribution (minority class >30%)
3. **Callbacks**: ModelCheckpoint monitors validation loss to save best model
4. **Evaluation**: Multi-metric analysis including confusion matrix and classification report

## 📝 Usage

Refer to the implementation notebook for step-by-step code execution. The notebook includes:
- Dataset loading and exploration
- Data augmentation pipeline
- Model building and training
- Performance evaluation
- Prediction on custom images

## 🔍 Key Learnings

- Transfer learning effectiveness for face recognition tasks
- Importance of data augmentation for model robustness
- Model checkpoint strategies for optimal model selection
- Comprehensive evaluation beyond simple accuracy metrics

---

*This project was developed to demonstrate deep learning capabilities in computer vision and binary classification tasks.*