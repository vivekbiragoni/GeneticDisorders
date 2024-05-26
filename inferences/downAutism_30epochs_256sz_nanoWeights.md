

## YOLOv8 Classification Model Training Documentation yolov8n-cls.pt

### Overview

This document summarizes the process and results of training a YOLOv8 classification model using Ultralytics HUB. The model was trained on a custom dataset and evaluated over 30 epochs.

### Environment and Setup

- **Ultralytics YOLOv8 Version:** 8.2.22
- **Python Version:** 3.10.12
- **Torch Version:** 2.3.0+cu121
- **CUDA Device:** Tesla T4 (15,102MiB)

### Training Configuration

- **Task:** Classification
- **Model:** yolov8n-cls.pt (pretrained)
- **Data Source:** [Dataset](https://storage.googleapis.com/ultralytics-hub.appspot.com/users/7lq1hDvjg3d6SrjjPN7xkSZdixs2/datasets/95NbOFzSbfe0ElCKkvh2/GD_out.zip)
- **Epochs:** 30
- **Image Size:** 256
- **Batch Size:** 16 (AutoBatch computed optimal size)
- **Optimizer:** AdamW (determined automatically)
- **Learning Rate (lr0):** 0.000714
- **Momentum:** 0.9
- **Workers:** 8
- **Mixed Precision:** AMP enabled

### Dataset Details


- **Train Set:** 4117 images
- **Validation Set:** 1175 images
- **Test Set:** 590 images
- **Classes:** 4

### Training Logs

#### Epoch-wise Performance

| Epoch | Top-1 Accuracy | Top-5 Accuracy | Train Loss | Validation Loss |
|-------|----------------|----------------|------------|-----------------|
| 1     | 0.423          | 0.958          | 1.098      | 1.107           |
| ...   | ...            | ...            | ...        | ...             |
| 21    | 0.871          | 1.000          | 0.212      | 0.881           |
| ...   | ...            | ...            | ...        | ...             |
| 30    | 0.886          | 1.000          | 0.17     | 0.877             |

#### Loss and Accuracy Trends

- **Training Loss:** Decreases steadily over epochs, indicating proper model learning.
- **Validation Loss:** Also decreasing, though fluctuations may occur due to model evaluation on validation data.
- **Top-1 Accuracy:** Improved from 42.3% to 87.1% by epoch 21.
- **Top-5 Accuracy:** Achieved 100% consistently after a few epochs.

### Key Observations

- **Optimizer and Learning Rate:** The AdamW optimizer with an initial learning rate of 0.000714 was used, providing a good balance between convergence speed and stability.
- **Batch Size Adjustment:** AutoBatch computed the optimal batch size as 16, ensuring efficient utilization of GPU memory.
- **Automatic Mixed Precision (AMP):** Enabled to enhance training efficiency and reduce memory usage.
- **Early Stopping and Patience:** The patience parameter was set to 100, meaning the training will continue up to 100 epochs without improvement before stopping early.

