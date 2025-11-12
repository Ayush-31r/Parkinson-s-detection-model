# Parkinson’s Disease Detection using CNNs

This project explores the detection of **Parkinson’s Disease** from voice recordings by converting audio signals into **spectrograms** and training a **Convolutional Neural Network (CNN)** with PyTorch.

---

## 🔍 Overview
- **Goal:** Detect Parkinson’s Disease from patient speech data.  
- **Approach:**
  - Preprocess audio into spectrograms using **PyTorch audio tools**.  
  - Train a **CNN model** on spectrogram images.  
  - Evaluate performance with accuracy, precision, recall, and F1-score.  

---

## 🌐 Live Demo  
👉 [**Parkinson's Detector**](https://parkinson-s-detection-model-deeplearning.streamlit.app/)

---

## 📊 Methodology
1. **Data Preprocessing:** Convert raw audio into spectrograms for visual representation of frequency vs. time.  
2. **Model:** CNN designed to capture spatial features from spectrograms.  
3. **Training:** Optimized with standard techniques (Adam optimizer, early stopping).  
4. **Evaluation:** Compared predictions against ground truth labels.  

---

## 🖼 Sample Spectrograms
Here are two spectrograms used in the project (one from a Parkinson’s patient and one from a healthy control):

<p align="center">
  <img src="Feature-Classification/sample1.png" alt="Spectrogram 1" width="45%"/>
  <img src="Feature-Classification/sample2.png" alt="Spectrogram 2" width="45%"/>
</p>

---

## 🚀 Results
- CNN achieved promising performance in distinguishing Parkinson’s patients from controls using voice-based spectrograms.  
- Demonstrates the potential of **deep learning in medical diagnosis from speech**.  

---

## ⚡ Tech Stack
- **Python**, **PyTorch** (for audio + deep learning)  
- **Matplotlib** (for visualization)  
- **Scikit-learn** (for evaluation metrics)  

---

## 📌 Future Work
- Explore **transfer learning** with pre-trained CNNs.  
- Improve robustness with **data augmentation**.  
- Extend dataset with more diverse samples for better generalization.  

