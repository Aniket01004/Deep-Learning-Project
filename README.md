# Deep-Learning-Project

COMPANY: CODTECH IT SOLUTIONS

NAME: ANIKET BHOGE

INTERN ID:CT06DF1221

DOMAIN: DATA SCIENCE

DURATION: 6 WEEKS

MENTOR: NEELA SANTOSH

# 🐱🐶 Cat and Dog Image Classification using TensorFlow

This project is part of the **CODTECH Virtual Internship in Data Science (Task 2)** and focuses on building a **Convolutional Neural Network (CNN)** model using **TensorFlow/Keras** to classify images of cats and dogs. The model is trained from scratch using a clean dataset and evaluated with custom input images to test real-world accuracy.

---

## 🚀 Project Objective

The main objective of this task is to demonstrate the ability to preprocess image datasets, build a deep learning model, and use it to classify binary image data (cats vs. dogs). This project replicates a common real-world image classification workflow in machine learning and computer vision.

The project includes training, evaluation, and testing with custom images. It's designed to help learners understand the pipeline from raw image data to a fully functional classification model.

---

## 🛠 Tools and Libraries Used

- **Python**: Core programming language
- **TensorFlow/Keras**: For building and training the CNN model
- **Matplotlib**: For plotting training accuracy and loss
- **Jupyter Notebook**: For step-by-step development

---


## 📁 Dataset Information

We use the following dataset:

🔗 [Dogs vs. Cats | Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats/data)

The dataset includes:
- `train/` folder with cat and dog images (e.g. `cat.0.jpg`, `dog.0.jpg`)
- `test1/` folder for prediction images (e.g. `1.jpg`, `2.jpg`)

> ⚠️ The dataset is not included in this repository due to size limitations.

### 📥 How to Use the Dataset

1. Download the dataset from Kaggle.
2. Extract it and organize it like:

```
cats-dogs-kaggle/
├── train/
│   ├── cat.0.jpg
│   ├── dog.0.jpg
│   └── ...
└── test1/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

---

## 💡 Project Workflow

1. **Data Loading & Preprocessing** using `ImageDataGenerator`
2. **CNN Model Building** with Conv2D, MaxPooling2D, Dense layers
3. **Model Training** with validation accuracy tracking
4. **Model Evaluation** using test data
5. **Custom Image Prediction** support

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/cats-vs-dogs-kaggle.git
cd cats-vs-dogs-kaggle
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download and place the dataset as described above.

4. Launch the notebook:

```bash
jupyter notebook Cats_Dogs_Kaggle_Classifier.ipynb
```

---

## 🎯 Example Prediction

```python
predict_custom_image("test1/200.jpg")
```

Output:

```
Predicted: Dog (0.91)
```










## 🧪 Features

- CNN-based binary classification (Cat vs Dog)
- Works with small image dataset
- Fast training (under 1 minute on CPU)
- Includes code to test on custom images

---

## 📚 Learning Outcomes

Through this project, you'll gain hands-on experience with:

- Building a CNN in TensorFlow
- Preprocessing image datasets using `ImageDataGenerator`
- Visualizing model performance with Matplotlib
- Deploying a lightweight model for quick testing

---
