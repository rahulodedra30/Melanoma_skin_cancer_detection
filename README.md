# Melanoma Skin Cancer Early Detection using TensorFlow

## Project Description

Human skin cancer is the most common and potentially life-threatening form of cancer, with melanoma exhibiting a particularly high mortality rate. Early detection is crucial for effective treatment. This project introduces a computer-aided detection technique for early melanoma diagnosis using Convolutional Neural Networks (CNNs). By leveraging deep learning and image processing techniques, this approach aims to detect melanoma from dermoscopic images with high accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Results](#results)
- [Logging and Monitoring](#logging-and-monitoring)
- [Conclusion](#conclusion)

  
## Introduction

Skin plays a crucial role in protecting us from environmental factors, but it is also susceptible to various malignancies, including melanoma. Melanoma is a severe form of skin cancer that, if detected early, can be effectively treated. This project focuses on developing a deep learning-based method to detect melanoma from dermoscopic images.

### Objectives

- Develop a CNN model to classify skin lesions as melanoma or non-melanoma.
- Use MLFlow for experiment tracking and logging.


## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rahulodedra30/Melanoma_skin_cancer_detection.git
   cd Melanoma_skin_cancer_detection


2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt


3. **Set up MLFlow:**

   
## Results
**Model Summary**

```
Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d_3 (Conv2D)               │ (None, 148, 148, 32)   │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_3 (MaxPooling2D)  │ (None, 74, 74, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_4 (Conv2D)               │ (None, 72, 72, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_4 (MaxPooling2D)  │ (None, 36, 36, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_5 (Conv2D)               │ (None, 34, 34, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_5 (MaxPooling2D)  │ (None, 17, 17, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_1 (Flatten)             │ (None, 36992)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 128)            │     4,735,104 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 1)              │           129 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 4,828,481 (18.42 MB)
 Trainable params: 4,828,481 (18.42 MB)
 Non-trainable params: 0 (0.00 B)
```


## Performance Metrics**

**Training Accuracy:** 91% <br>
**Validation Accuracy:** 90%<br>
**Test Accuracy:** 91%<br>

## Logging and Monitoring
This project uses MLFlow for tracking experiments, logging metrics, and storing models. Key logs include:

**Model parameters**
Training and validation metrics<br>
Test accuracy<br>
Model artifacts (saved models)<br>

## Conclusion
The CNN model demonstrated high accuracy in detecting melanoma, surpassing traditional methods.




