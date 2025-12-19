# SMS Spam Detection using NLP & Machine Learning

## Project Overview
The automatic detection of spam messages is a key challenge in modern communication systems.  
This project implements a complete **machine learning pipeline** to classify SMS messages as **Spam** or **Ham** using **Natural Language Processing (NLP)** techniques and supervised learning models.

The focus is on **text preprocessing**, **feature extraction**, **dimensionality reduction**, and **model evaluation**, with an emphasis on comparing multiple classifiers.

> **Note**: The notebook and in-code comments are written in French, as this project was completed in a French academic context.

---

## Dataset
- **Name**: SMS Spam Collection Dataset  
- **Source**: UCI Machine Learning Repository  
- **Size**: 5,574 SMS messages  
- **Target variable**:  
  - `spam`  
  - `ham`

The dataset consists of short text messages labeled as spam or legitimate messages.

---

## Methodology

### 1. Text Preprocessing
The following preprocessing steps were applied:
- Lowercasing text  
- Removal of punctuation and special characters  
- Tokenization  
- Stopwords removal  
- Stemming  

### 2. Feature Extraction
- Text vectorization using **TF-IDF**
- Dimensionality reduction using a **Neural Network Autoencoder** implemented with **Keras**

### 3. Classification Models
Several supervised learning models were trained and compared:
- Logistic Regression  
- Support Vector Machine (SVM)  
- Multi-Layer Perceptron (MLP)  

### 4. Model Evaluation
Model performance was evaluated on a test set using:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## Results
- All models achieved strong overall performance for spam detection.
- **SVM and MLP** showed particularly good recall for the spam class.
- The use of **autoencoder-based embeddings** provided a dense representation of text features; however, traditional **TF-IDF** representations remained highly competitive for this dataset.

These results highlight the importance of selecting feature representations adapted to the characteristics of the data.

---

## Tech Stack
- **Programming Language**: Python  
- **Libraries**:  
  - Scikit-learn  
  - TensorFlow / Keras  
  - NLTK  
  - Pandas, NumPy  
- **Environment**: Jupyter Notebook  

---

## Repository Structure
