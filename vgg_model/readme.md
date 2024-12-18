# **Waste Classification Project (Resikel)**

This project implements a **waste classification feature** for the **Resikel** application. It uses a **VGG-based Convolutional Neural Network (CNN)** to classify images of waste into predefined categories and predict recycling recommendations.

---

## **Key Features**

### **Image Classification**
- Classifies waste images into **5 categories**:  
   - **Botol Kaca**  
   - **Kaleng**  
   - **Kardus**  
   - **Kertas**  
   - **Plastik**  

### **Dataset**
- The dataset contains a total of **6000+ images** split into:  
   - **Training Set**: 70% of the data  
   - **Testing Set**: 20% of the data  
   - **Validation Set**: 10% of the data  

### **Model Architecture**
- Implements a **VGG-based CNN** for robust image classification.  
- Architecture includes:
   - Convolutional layers with ReLU activations.  
   - Max-pooling layers for feature downsampling.  
   - Fully connected layers for final predictions.  

### **Image Preprocessing**
- Resizes input images to **224x224** pixels (compatible with VGG).  
- Applies normalization to scale pixel values to a range of 0–1.  
- Augments data with:  
   - Horizontal flips  
   - Rotations (±15°)  
   - Cropping (zoom range: 0–20%)  
   - Grayscale (15% probability)  
   - Blur up to 2.5px  

### **Training and Evaluation**
- The model is trained on the preprocessed and augmented dataset.  
- **Evaluation Metrics**:
   - Accuracy  
   - Precision  
   - Recall  

### **Output Interface**
- A simple example **web interface** allows users to upload an image of waste and receive predictions:  
   - **Class Label**: e.g., "Botol Kaca"  
   - **Confidence Score**: e.g., "92.91%"  

---

## **Tech Stack**
- **Backend**: Flask (Python Web Framework)  
- **Machine Learning**: TensorFlow/Keras  
- **Model**: VGG-based Convolutional Neural Network  
- **Development**: Jupyter Notebook  
- **Deployment**: Docker (Optional)

---

## **Requirements**
- Python 3.10+  
- TensorFlow/Keras  
- Flask  
- NumPy, OpenCV, Matplotlib 
