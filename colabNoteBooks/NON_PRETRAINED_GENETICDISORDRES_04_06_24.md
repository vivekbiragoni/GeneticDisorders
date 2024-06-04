

## Project Overview

The project involves training a YOLOv8n-cls model for classifying images into one of four classes. The model architecture consists of 99 layers and approximately 1,443,412 parameters. The training process utilizes a dataset comprising 4117 training images, 1175 validation images, and 590 testing images, all belonging to the same four classes.

## Environment Setup

- **Python Version:** 3.10.12
- **Torch Version:** 2.3.0+cu121
- **GPU Used:** Tesla T4 (15102MiB)
- **Total GPU Memory:** 14.75G
- **Reserved Memory:** 0.22G
- **Allocated Memory:** 0.04G
- **Free Memory:** 14.49G

## Model Configuration

- **Input Size:** 256x256 pixels
- **Number of Classes (nc):** Customized from 1000 to 4
- **Backbone:** Utilizes Conv layers and C2f blocks for feature extraction, followed by a `Classify` head layer with an output size matching the number of classes.
- **Optimizer:** AdamW, with automatic selection of learning rate (`lr0`) and momentum, tailored to different parameter groups.

## Training Process

- **Dataset Preparation:** Images were found in the specified directories without any corruption.
- **Model File Download:** The base YOLOv8n model weights were downloaded from Ultralytics GitHub assets.
- **Automatic Mixed Precision (AMP):** Enabled to optimize training speed without compromising performance.
- **Optimal Batch Size Determination:** Due to a CUDA anomaly, the batch size was automatically set to 16, which was within the available GPU memory limits.
- **Runtime Warning:** A warning regarding the compatibility of `os.fork()` with multithreaded code was issued, potentially leading to deadlocks.
- **Training Duration:** Completed 30 epochs in approximately 0.394 hours.
- **Performance Metrics:** Achieved a top-1 accuracy of 77.7% on the validation set.
- **Model Saving:** Both the last and best models were saved, with the optimizer stripped from the files to reduce their size.

## Validation and Testing

- **Validation Accuracy:** Improved to 77.7%, indicating good model performance on unseen data.
- **Inference Speed:** The model demonstrated fast preprocessing and inference times, contributing to efficient processing of images.

## Model Deployment

- **Model Syncing:** The trained model was synced to the Ultralytics HUB, making it accessible for further use and sharing.

