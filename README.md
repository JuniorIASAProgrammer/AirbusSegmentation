# AirBus Ship Segmentation
Customized implementation of the U-Net in Tensorflow for Kaggle's [Airbus Ship Detection Challenge](https://www.kaggle.com/competitions/airbus-ship-detection) from high definition images.

## Quick start

1. Clone repository

2. Install dependencies
```bash
pip install -r requirements.txt
```

## **Usage**
Note : Use Python 3.6 or newer

## Training

```console
> python train.py -h
usage: train.py [-h] [--epochs] [--batch-size] [--learning-rate]
                [--image-path] [--label-path] [--save-model-path] [--augmentation]

Train the U-Net on images

options:
  -h, --help            show this help message and exit
  --image-path          Folder with train images
  --label-path          Path to .csv file
  --epochs              Number of epochs
  --batch-size          Batch size
  --learning-rate       Learning rate
  --save-model-path     Path to save fitted model
  --augmentation        Apply data augmentation or not
```

## Evaluate

```console
> python evaluate.py 
usage: evaluate.py [-h] [--image-path] [--batch-size] [--model-path]
                [--binary-mask] [--threshold]

Predict masks with U-Net on images

options:
  -h, --help            show this help message and exit
  --image-path          Folder with train images
  --batch-size          Batch size
  --model-path          Path to model (default: pretrained)
  --binary-mask         Plot class prediction (True) or probabilistic prediction (False)
  --threshold           Threshhold for pixel binary classification (only if --binary-mask is True)
```