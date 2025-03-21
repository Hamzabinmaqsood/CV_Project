# 🔧 AI-Powered Fault Detection from Acoustic Data  
## 🎧 Deep Learning for Predictive Maintenance

# Table of Contents

[Project Overview](#Project-Overview)

1. [Abstract](#Abstract)
2. [Introduction](#introduction)  
3. [Installation](#Installation)
4. [Methodology](#methodology)
5. [Data Augmentation](#Data-Augmentation)
6. [Model Training](#model-training)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results](#results)
9. [Future Improvements](#future-improvements)
10. [Conclusion](#conclusion)

## 📝 Project Overview  
Industrial machines generate specific sounds when they operate, and deviations in these sounds can indicate faults.  
However, traditional fault detection methods rely on **manual inspections** or **vibration-based monitoring**, which are **expensive and time-consuming**.  

This project presents an **AI-powered, non-invasive approach** using **deep learning** to analyze acoustic signals.  
The process involves:  
🔹 **Transforming sound into images** using Short-Time Fourier Transform (STFT) and Mel-Frequency Cepstral Coefficients (MFCCs).  
🔹 **Training a Convolutional Neural Network (CNN)** on these image representations to classify different types of machine faults.  
🔹 **Achieving high accuracy** in fault classification, making it an efficient and reliable solution for predictive maintenance.  


## Abstract
Early fault detection in industrial machinery is critical for minimizing downtime and maintenance expenses. Traditional inspection methods, such as vibration monitoring or manual checks, often demand specialized hardware and expertise. This project demonstrates an automated approach that analyzes acoustic signals with deep learning. Sound data is first converted into images through techniques like Short-Time Fourier Transform (STFT) and Mel-Frequency Cepstral Coefficients (MFCCs). A convolutional neural network (CNN) is then trained on these image representations to classify various types of faults. Achieving high accuracy, the proposed method provides a reliable and efficient solution for automated fault detection in industrial contexts. Future work may include refining the feature extraction process and deploying the system for real-time monitoring.

## Introduction
In industrial environments, promptly identifying machine faults is essential to maintain productivity and avoid costly failures. Machines typically exhibit unique sound patterns when faults emerge. Conventional methods (for instance, vibration analysis) can be expensive and require specialized knowledge. This project presents an automated system that transforms audio signals into images and applies a CNN to detect different fault types. The primary objective is to build a fast, accurate, and reliable fault detection tool suitable for real-world industrial use.

## Installation
To use the project, follow these steps:

1. Clone the Repository
Clone the GitHub repository to your local machine.

``` git clone https://github.com/muneebashafiq/Automatic-machine-fault-detection-from-Acoustic-Data-using-Deep-Learning ```

2. Install Dependencies
Install the required dependencies by running the following command in your terminal.
```
pip install -r requirements.txt
```

Usage
To use the project:

1. Run on Google Colab:
Upload your dataset to Google Drive.
Mount your drive in Colab:
```
from google.colab import drive
drive.mount('/content/drive')
```

Set the dataset path accordingly:
```
audio_dir = '/content/drive/MyDrive/dataset_folder/'
```

2. Running Locally
To run the project locally, follow these steps:

Run the Project: Start the application.
```
python Machine_fault_detection.ipynb
```
Or Use Jupyter Notebook
Dataset

Download the Dataset form this link:
```
https://drive.google.com/drive/folders/1Y7-NCyYQb0KRe6BtK8PycGHIWrHkDZ_L
```

## Methodology
This project follows a methodical workflow: data preprocessing, feature extraction, data augmentation, model training, and evaluation.
#Data Preprocessing
Audio signals captured from operating machines are standardized and segmented. Rather than handling long audio files, each recording is divided into 0.2-second clips with a 20% overlap. This ensures every snippet captures meaningful sound information and maintains continuity. The sampling rate of 44.1 kHz is chosen to preserve audio quality.
#Feature Extraction
To convert audio segments into images suitable for CNNs, several feature extraction methods are used:
STFT Magnitude Images: These visualize how sound frequencies vary over time. By splitting the audio signal into small time windows, STFT highlights frequency changes and intensities that may indicate faults.
STFT Phase Images: While magnitude focuses on the intensity of frequencies, phase images capture the timing and directional attributes of sound waves. Phase patterns can shift unpredictably when a machine fault occurs, offering additional insight beyond magnitude alone.
MFCCs: Mel-Frequency Cepstral Coefficients emulate how the human ear perceives sounds. They distill key audio characteristics while filtering out irrelevant background noise, making them especially useful for identifying fault-related changes.
All these features are turned into images to allow CNN-based analysis.

## Data Augmentation
To bolster the model’s robustness, various augmentations are applied:
Time Stretching: Speeds up or slows down the audio without changing pitch, enabling the model to recognize the same fault at different operational speeds.
Pitch Shifting: Alters pitch while maintaining speed, accounting for slight variations caused by factors like temperature or load changes.
Noise Addition: Introduces background noise to mirror real industrial settings, making the model more resilient to environmental interference.
These techniques expand the dataset’s diversity and enhance generalization.

## Model Training
A convolutional neural network (CNN) classifies the spectrogram images. The dataset is split into 80% for training and 20% for testing. The CNN structure includes:
Convolutional Layers: Identify edges, textures, and patterns in the images, capturing the unique acoustic signatures of faults.
Max Pooling Layers: Reduce image dimensions while preserving critical features, optimizing computational efficiency.
Fully Connected Layers: Aggregate extracted features to classify various fault types accurately.
Dropout Layers: Randomly disable neurons during training to avoid overfitting, improving the model’s ability to generalize.
Softmax Output Layer: Outputs the probability distribution across fault categories, selecting the most likely fault type.

## Evaluation Metrics
Several metrics are used to assess the model:
Accuracy: The percentage of correct predictions over total samples.
Loss Graphs: Track how well the model learns across epochs—lower loss signifies better training progress.
Confusion Matrix: Offers a detailed view of which classes are correctly or incorrectly identified, helping spot categories the model finds challenging.
Comparison with Other Models: The CNN’s performance is evaluated against alternative architectures to ensure optimal results.
The final model reaches about 87% accuracy, highlighting its capability to detect faults from audio signals.

Model Pipeline & Results 
This project presents an automated system for detecting machine faults through audio analysis and deep learning. By converting audio snippets into image representations and training a CNN, it accurately classifies different types of faults. This approach is both reliable and efficient, laying the groundwork for practical applications in real industrial settings.

## Results
### 🔹 **Model Architecture Overview**
<img src="https://github.com/user-attachments/assets/9904042c-f29c-4156-bae5-092abf2bb155" width="600">
<img src="https://github.com/user-attachments/assets/3f4c8948-7122-401b-89e8-d85e64f72e37" width="600">
 
### 🔹 **Model Training Progress**  
<img src="https://github.com/user-attachments/assets/9c24b9b3-b995-4668-b20c-e0364f803f37" width="600">


### 🔹 **Model Evaluation Results**  
<img src="https://github.com/user-attachments/assets/18a48c5d-6c12-4cec-8aac-0cc8fa92e952" width="600">
 

### 🔹 **Confusion Matrix for Model Performance**  
<img src="https://github.com/user-attachments/assets/cb6215d5-5d83-4fe5-b726-9812a6f60a26" width="600">


There are opportunities for further improvements:

Fine-Tuning Hyper-parameters: Adjusting learning rates, batch sizes, and network layers can optimize performance. This ensures better training efficiency and improves fault detection accuracy.
Enhancing Feature Extraction Methods: Exploring advanced sound processing techniques can help capture more meaningful features. This may improve classification accuracy and make the model more robust.
Deploying in Real-Time Applications: Integrating the model into an industrial system for continuous monitoring is essential. This will allow real-time fault detection, preventing potential failures before they occur.
By implementing these improvements, the system can become even more robust and practical for industrial use.

##  Key Features  
- 🎤 **Non-invasive fault detection** using sound analysis  
- 🖼 **Image-based approach** with STFT and MFCC feature extraction  
- 🤖 **Deep Learning (CNN) for classification** of machine faults  
- 📊 **High accuracy** achieved through optimal model tuning  
- 🏭 **Potential for real-time deployment** in industrial environments  


## Conclusion
This project presents a computer vision-based approach for detecting machine faults using sound signals. By converting audio data into spectrogram images and using a CNN model, an efficient and automated fault detection system is developed. The model achieves high accuracy, demonstrating its effectiveness in identifying different fault types. Future work will focus on refining the model and deploying it in real-world industrial applications, making fault detection faster, more reliable, and accessible to industries worldwide.
