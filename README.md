# Chatbot and Waste Classification Project (Resikel)

This repository contains two key components for an intelligent waste management system: a **Chatbot for Recycling Inquiries and Swearing Detection** and a **Waste Image Classification System**. Together, these systems aim to enhance user engagement and facilitate proper waste management using advanced machine learning and deep learning techniques.

---

## **1. Chatbot Project (AskResikel)**

### **Overview**

The chatbot is designed to classify user inputs into predefined intents, such as recycling-related questions or inappropriate language detection, and provide automated responses.

### **Key Features**

- **Intent Recognition**:  
  - Recognizes and classifies user inputs into intents such as:
    - Recycling inquiries
    - Swearing/inappropriate language detection  
  - Utilizes **TF-IDF** for text vectorization and **Support Vector Machines (SVM)** for classification.

- **Response Generation**:  
  - Predefined responses for each intent.
  - If the input has low similarity to existing patterns, the chatbot prompts for clarification.

- **API Integration**:  
  - Provides a RESTful API endpoint (`/chat`) using **Flask** and **Flask-RESTful**.
  - Handles user inputs via **POST** requests and returns JSON responses.

- **Evaluation and Performance**:  
  - Model evaluation using:
    - Confusion Matrix
    - Classification Report (accuracy, precision, recall)

- **Web Interface**:  
  - Simple browser-based interface for interacting with the chatbot.

- **Docker Support**:  
  - Dockerized for easy deployment in any environment.

### **Workflow**

1. **User Input**:  
   User asks: _"What should I do with plastic bottles?"_
2. **Intent Classification**:  
   The chatbot identifies the intent as "recycle".
3. **Response Generation**:  
   Chatbot responds: _"You should recycle plastic bottles by placing them in the designated recycling bin."_
4. **Clarification**:  
   If unclear: _"Sorry, I didn't understand. Could you please rephrase your question?"_

---

## **2. Waste Classification Project (Resikel)**

### **Overview**

This component implements an image classification system that predicts the type of waste from uploaded images and provides recycling recommendations. It is based on a **VGG-based Convolutional Neural Network (CNN)**.

### **Key Features**

- **Image Classification**:  
  Classifies images into **5 categories** of waste:
  1. **Botol Kaca**  
  2. **Kaleng**  
  3. **Kardus**  
  4. **Kertas**  
  5. **Plastik**

- **Dataset**:  
  - Contains over **6000+ images**.
  - Split into:
    - **Training Set**: 70%  
    - **Testing Set**: 20%  
    - **Validation Set**: 10%

- **Model Architecture**:  
  - Implements a **VGG-based CNN** for robust image classification.  
  - Architecture includes:
    - Convolutional Layers with ReLU activation.  
    - Max-Pooling Layers for feature downsampling.  
    - Fully Connected Layers for classification.  

- **Image Preprocessing**:  
  - Images are resized to **224x224** pixels.
  - Pixel values are normalized to a range of **0–1**.  
  - Data Augmentation includes:
    - **Horizontal Flip**
    - **Rotation** (±15°)  
    - **Cropping** (zoom range: 0–20%)  
    - **Grayscale** (15% probability)  
    - **Blur** (up to 2.5px)

- **Training and Evaluation**:  
  - **Model Evaluation Metrics**:  
    - Accuracy  
    - Precision  
    - Recall  

- **Output Interface**:  
  - Users can upload an image via the web interface and receive:  
    - **Predicted Class Label**: e.g., "Botol Kaca"  
    - **Confidence Score**: e.g., "92.91%"  

- **Docker Support**:  
  - Application can be containerized and deployed using Docker.

---

## **Tech Stack**

### **Chatbot**
- **Backend**: Flask (Python Web Framework)
- **Machine Learning**: TF-IDF + SVM (Support Vector Classifier)
- **API**: Flask-RESTful
- **Containerization**: Docker  

### **Waste Classification**
- **Backend**: Flask (Python Web Framework)
- **Machine Learning**: TensorFlow/Keras
- **Model**: VGG-based CNN
- **Image Processing**: OpenCV, NumPy
- **Development**: Jupyter Notebook
- **Containerization**: Docker

---

## **Example Use Cases**

1. **Chatbot for Recycling**:  
   - User: "How do I recycle glass bottles?"  
   - Chatbot: "Place glass bottles in a separate bin for glass recycling."

2. **Chatbot for Swearing Detection**:  
   - User: _inappropriate input_  
   - Chatbot: "Please use respectful language."

3. **Waste Image Classification**:  
   - User uploads an image of a **plastic bottle**.  
   - Prediction:  
     - **Class**: "Plastik"  
     - **Confidence Score**: "95.6%"

---

## **Requirements**

- **Python**: 3.10+  
- **Libraries**:  
  - Flask  
  - Flask-RESTful  
  - TensorFlow/Keras  
  - NumPy, OpenCV, Matplotlib  
  - Scikit-learn  
- **Docker** (Optional for deployment)

---

## **How It Works**

### **Chatbot Workflow**  
1. Preprocess user input with **TF-IDF**.
2. Predict intent using **SVM**.
3. Match patterns using **cosine similarity**.
4. Generate appropriate responses.

### **Waste Classification Workflow**  
1. Preprocess input image (resize, normalize).  
2. Apply the trained **VGG-based CNN** model.  
3. Output the waste classification result and confidence score.
