# Bone Fracture Detection and Classification

This repository contains a comprehensive pipeline for detecting and classifying bone fractures using deep learning. The implementation is designed for ease of use, scalability, and interpretability, and it includes a user-friendly Streamlit app for real-time predictions.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset Folder Structure](#dataset-folder-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running the Streamlit App](#running-the-streamlit-app)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

Bone fractures are a critical medical condition requiring timely and accurate diagnosis. This project provides an automated system to identify fractures from X-ray images. It incorporates:
- A TensorFlow-based deep learning model for fracture classification.
- Pre-trained MobileNetV3 for efficient feature extraction.
- A user-friendly Streamlit application for interactive usage.

---

## Features
- **Preprocessing**: Includes corrupted image detection and removal, as well as data augmentation.
- **Transfer Learning**: Utilizes MobileNetV3 for efficient and accurate classification.
- **Metrics and Reporting**: Provides detailed metrics such as Accuracy, Precision, Recall, and AUC.
- **Streamlit App**: An interactive web application for real-time image classification.
- **Extensibility**: Modular codebase for easy customization and scalability.

---

## Dataset Folder Structure

Organize the dataset as follows:

```
datasets/
├── train/
│   ├── fractured/
│   ├── non_fractured/
├── val/
│   ├── fractured/
│   ├── non_fractured/
├── test/
    ├── fractured/
    ├── non_fractured/
```

Each subfolder contains images corresponding to the labeled categories.

## Dataset Link

- https://drive.google.com/file/d/1WeuxOenviI1_ElW5ISED4MhvR_YFYdmB/view

Unzip the original folder and copy the train, test and val folders into the datasets folder

---

## Installation

### Prerequisites
Ensure you have Python 3.7 or above installed.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bone-fracture-detector.git
   cd bone-fracture-detector
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify the dataset is correctly organized in the `datasets/` directory.

---

## Usage

### Training the Model
Run the `Bone_Fracture_Model_Generator_Script.py` to preprocess the data and train the model:
```bash
python Bone_Fracture_Model_Generator_Script.py
```

### Running the Streamlit App
The Streamlit app provides a web interface for classifying new images.

1. Launch the app:
   ```bash
   streamlit run Bone_Fracture_App.py
   ```

2. Access the app in your browser at `http://localhost:8501`.

3. Upload an X-ray image, and the app will classify it as either `fractured` or `non_fractured`.

---

## Future Enhancements
- Add support for additional medical image modalities.
- Include explainability features such as Grad-CAM for visualizing model predictions.
- Optimize the model for deployment on edge devices.

---
