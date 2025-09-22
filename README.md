# 🧠 Alzheimer’s Disease Classification using Deep Learning  

## 📌 Project Overview  
This project presents a **deep learning-based approach** for the classification of **Alzheimer’s Disease (AD) stages** from **MRI brain images**. The motivation behind this work is to contribute toward **early detection and diagnosis** of Alzheimer’s, which is critical in improving treatment planning and patient care.  

Due to the challenge of **imbalanced datasets** in medical imaging, this work integrates advanced preprocessing techniques such as **image augmentation** and **oversampling**. Two state-of-the-art convolutional neural networks (CNNs) – **ResNet** and **DenseNet** – were implemented and evaluated.  

---

## 🎯 Objectives  
- Develop an accurate and reliable deep learning model for Alzheimer’s classification.  
- Address the issue of **class imbalance** in MRI datasets through preprocessing.  
- Evaluate model performance using **standard classification metrics**.  
- Lay the foundation for **future research** in explainable AI and clinical deployment.  

---

## 🛠️ Technologies Used  
- **Programming Language:** Python  
- **Libraries & Frameworks:**  
  - Deep Learning: TensorFlow, Keras  
  - Computer Vision: OpenCV  
  - Data Handling: NumPy, Pandas  
  - Visualization: Matplotlib, Seaborn  
- **Models Implemented:** ResNet  

---
## 📂 Dataset

This project is based on the **Alzheimer’s 4-Class Dataset** originally published on Kaggle:  
[Alzheimer 4 Class Dataset](https://www.kaggle.com/datasets/preetpalsingh25/alzheimers-dataset-4-class-of-images)

For experimentation, I created a **modified version of this dataset** to handle class imbalance and improve training performance.
[Augmented Alzheimer Dataset](https://www.kaggle.com/datasets/samiulhaq45/final-alzheimer-dataset)

⚠️ Due to size restrictions, the dataset may not be directly available in this GitHub repository.  
You can download the original dataset from Kaggle using the links above and also the modified version.
---

## ⚙️ Methodology  

### 1. Data Preprocessing  
- Input dataset: MRI brain images.  
- Challenges: Highly **imbalanced classes** (more normal samples than disease samples).  
- Applied techniques:  
  - **Image Augmentation** (rotation, flipping, zooming).  
  - **Oversampling** to balance dataset distribution.  

### 2. Model Development  
- Implemented **ResNet** architectures.  
- Fine-tuned hyperparameters (learning rate, batch size, epochs).  
- Trained models on augmented datasets.  

### 3. Evaluation Metrics  
- Accuracy  
- Precision, Recall, F1-score  
- Confusion Matrix  

---

## 📊 Results  

- **ResNet Model:** Achieved **96% accuracy** on the test set.

---

## 🚀 Future Work  

- Integrating **Explainable AI (XAI)** techniques (e.g., Grad-CAM, LIME) to make predictions interpretable for clinicians.  
- Experimenting with advanced architectures like **EfficientNet** or **Vision Transformers (ViTs)**.  
- Deploying the model as a **web or mobile application** for practical healthcare usage.  
- Expanding the dataset for better generalization and robustness.   

