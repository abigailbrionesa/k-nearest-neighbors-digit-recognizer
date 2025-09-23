k-Nearest Neighbors Digit Recognizer – CSC 171 Project

Author: Abigail Briones
Course: CSC 171 – Introduction to Computer Science
Project: The Mystery of the Siwa Oasis Stones

Overview

This project implements a k-Nearest Neighbors (k-NN) classifier to recognize digits from digital images. Inspired by the tale of the mysterious Siwa Oasis stones, the goal is to predict the labels of partially eroded stone carvings using supervised learning techniques.

The project demonstrates:
Proficiency in algorithmic design and problem decomposition.
Modular and reusable Python code for real-world tasks.
A practical introduction to AI and machine learning concepts such as pattern recognition and similarity-based classification.
Thorough testing and debugging practices to ensure program correctness.

Project Motivation

Indiana Jones discovers ancient stones with faded digits, while Professor Paleontologos provides digital representations of some well-preserved carvings. These labeled stones serve as training data, while the eroded, unlabeled stones are test cases. Using k-NN, this project simulates how AI can “learn from examples” and make predictions for unseen data, bridging classical computer science with the foundations of artificial intelligence.

Features

Flatten Images
Converts 2D images (lists of lists of pixel values) into 1D lists for processing with distance functions.

Distance Computation
Implements both Manhattan and Euclidean distance metrics to measure similarity between images.

k-Nearest Neighbors Selection
Sorts distances and selects the k closest training images to a test image.

Majority Voting
Predicts labels based on the most frequent label among the k nearest neighbors, handling ties appropriately.

End-to-End Prediction
Integrates all steps to classify unknown digits accurately, demonstrating a working supervised learning pipeline.

Getting Started
Prerequisites

Python 3.8+

Recommended IDE: VS Code, PyCharm, or any Python environment

Installation

Clone the repository:

git clone https://github.com/yourusername/knn-digit-recognizer.git
cd knn-digit-recognizer

Usage

Place your training images in train_images.txt and test images in test_images.txt.

Run the classifier:

python main.py


Adjust k and distance metrics in main.py to explore performance variations.

Testing

This project emphasizes robust testing:

Unit tests for each function (flatten_image, manhattan, euclidean, k_nearest_neighbors, majority_vote)

Integration tests for the full prediction pipeline

Custom test cases with small, easy-to-visualize images ensure correctness

Design Decisions

Choice of k: Small values (1–3) ensure responsiveness to subtle differences, while larger k helps smooth out noise.

Distance metrics: Both Manhattan and Euclidean distances were implemented to study sensitivity to pixel intensity variations.

Modular design: Functions are reusable and maintainable, allowing easy extension to larger datasets such as MNIST.

Learning Outcomes

Through this project, I have:

Practiced modular software engineering by building reusable functions and clear program structure.

Gained hands-on experience with distance metrics and similarity-based learning.

Strengthened debugging, testing, and algorithmic problem-solving skills.

Explored foundational concepts in artificial intelligence, such as pattern recognition and supervised learning.

Future Work

Extend the classifier to handle larger, real-world datasets like MNIST.

Experiment with weighted k-NN or other distance metrics to improve accuracy.

Integrate image preprocessing techniques to reduce noise and enhance classification.

Contributing

Contributions, suggestions, and improvements are welcome. This project is a learning platform for exploring AI and algorithm design.

License

This project is licensed under the MIT License.


