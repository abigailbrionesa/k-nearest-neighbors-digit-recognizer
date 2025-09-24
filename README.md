# k-Nearest Neighbors Digit Recognizer – CSC 171 Project

**Author:** Abigail Briones  
**Course:** CSC 171 – Introduction to Computer Science  
**Project:** The Mystery of the Siwa Oasis Stones  

---

## Overview
A k-Nearest Neighbors (k-NN) classifier to recognize digits from digital images. Inspired by the tale of the Siwa Oasis stones, it predicts labels for partially eroded stone carvings using supervised learning.  

**Highlights:**  
- Algorithmic design and problem decomposition  
- Modular, reusable Python code  
- Introduction to AI concepts: pattern recognition, similarity-based classification  
- Thorough testing and debugging practices  

---

## Motivation
Indiana Jones finds ancient stones with faded digits. Labeled digital representations serve as training data, while eroded stones are test cases. This project simulates AI learning from examples to predict unseen data.  

---

## Features
- **Flatten Images:** Converts 2D images to 1D lists for processing  
- **Distance Computation:** Supports Manhattan and Euclidean metrics  
- **k-NN Selection:** Finds the k closest training images  
- **Majority Voting:** Predicts labels based on nearest neighbors  
- **End-to-End Prediction:** Full pipeline for digit classification  

---

## Getting Started

### Prerequisites
- Python
- Recommended IDE: VS Code, PyCharm, or any Python environment  

### Installation
```bash
git clone https://github.com/yourusername/knn-digit-recognizer.git
cd knn-digit-recognizer
