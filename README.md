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
- **TensorFlow & Keras**: For building and training the CNN model
- **Matplotlib**: For visualizing performance metrics
- **NumPy**: For numerical computations
- **OpenCV**: For image processing during testing
- **Jupyter Notebook**: To present and run the code interactively

---

## 📂 Project Structure

```
.
├── cats_and_dogs_filtered/     # Downloaded and extracted dataset folder
│   ├── train/
│   └── validation/
├── custom_image.jpg            # Custom image for prediction (optional)
├── cat_dog_classification.ipynb
└── README.md
```

- `cats_and_dogs_filtered/`: Dataset folder containing 'train' and 'validation' directories with 'cats' and 'dogs' subfolders.
- `cat_dog_classification.ipynb`: Jupyter Notebook containing the entire model-building, training, evaluation, and prediction pipeline.
- `custom_image.jpg`: You can add your own image for testing.
- `README.md`: This file, describing the project in detail.

---

## 🔄 Project Workflow Overview

### 1. Load Dataset  
Use `ImageDataGenerator` to load and normalize image data from the `cats_and_dogs_filtered` directory with proper labels.

### 2. Build the CNN  
Construct a simple CNN model using `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense` layers.

### 3. Compile & Train  
Use the `adam` optimizer and `binary_crossentropy` loss to train the model over multiple epochs.

### 4. Visualize Accuracy  
Plot training and validation accuracy curves to evaluate overfitting and performance.

### 5. Test on Custom Image  
Load your own image using OpenCV or Keras, resize it, and classify it using the trained model.

---

## 💡 Key Features

- Easy-to-follow CNN implementation using TensorFlow/Keras
- Works with real image datasets structured by folders
- Includes code to test on any custom cat/dog image
- Clean, modular, and beginner-friendly code
- Visual output of training progress and custom predictions

---

## 📌 How to Use

1. Download and extract the dataset from [this link](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip).
2. Place the extracted `cats_and_dogs_filtered` folder in the same directory as the notebook.
3. (Optional) Add a test image (e.g., `custom_image.jpg`) for prediction.
4. Open `cat_dog_classification.ipynb` in **Jupyter Notebook**, **Colab**, or **VS Code**.
5. Run all cells step by step to:
   - Load data
   - Build and train the model
   - Evaluate the model
   - Make predictions on new images

---

## 📚 Learning Outcomes

Through this project, I gained hands-on experience in:

- Loading and preprocessing real image datasets
- Designing CNN architectures from scratch
- Training, validating, and testing classification models
- Evaluating model performance visually and numerically
- Using TensorFlow and Keras for computer vision tasks

---

## 📧 Contact

Feel free to connect or reach out for suggestions, feedback, or collaboration:

**Name**: Aniket Bhoge  
**Email**: aniketbhoge04@gmail.com  
**LinkedIn**: [linkedin.com/in/aniketbhoge](https://www.linkedin.com/in/aniketbhoge)
