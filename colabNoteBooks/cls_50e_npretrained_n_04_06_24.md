# YOLOv8 Classification Model Training Notes

## Model Overview

- **Model Type:** YOLOv8n-cls
- **Purpose:** Classify images into one of four classes.
- **Model Architecture:**
  - Layers: 99
  - Parameters: 1,443,412
  - GFLOPs: 3.4

## Model Configuration

- **Input Size:** 256x256
- **Number of Classes (nc):** Overridden from 1000 to 4
- **Backbone:**
  - Conv layers and C2f blocks utilized for feature extraction.
  - Head: `Classify` layer with output size of 4 (matching the number of classes).
- **Optimizer:** AdamW with automatic selection of learning rate and momentum.
  - Initial learning rate (lr0): 0.000714
  - Momentum: 0.9
  - Weight Decay: Differentiated across parameter groups.

## Dataset

- **Training Data:**
  - Number of Images: 4117
  - Classes: 4
- **Validation Data:**
  - Number of Images: 1175
  - Classes: 4
- **Testing Data:**
  - Number of Images: 590
  - Classes: 4

## Training Details

- **Batch Size:** 16 (automatically determined based on GPU memory)
- **Epochs:** 50
- **Mixed Precision Training (AMP):** Enabled and successfully verified
- **Training Time:** 0.533 hours

## Results

- **Validation Metrics:**
  - Top-1 Accuracy: 82.2%
  - Top-5 Accuracy: 100%
- **Inference Speed:**
  - Preprocessing: 0.1ms per image
  - Inference: 0.6ms per image
  - Loss Calculation: 0.0ms per image
  - Postprocessing: 0.0ms per image

## Model Files

- **Best Model:** `runs/classify/train/weights/best.pt` (3.0MB)
- **Last Model:** `runs/classify/train/weights/last.pt` (3.0MB)

## TensorBoard

- **Visualization:** Model graph visualization added.
- **TensorBoard Command:** `tensorboard --logdir runs/classify/train`

## Repository Sync

- **Model Synced to Ultralytics HUB:** [Link to Model](https://hub.ultralytics.com/models/szeffWoo6JsaWHYyqulX)

ar understanding of the model settings and the achieved results for further experimentation and usage.
