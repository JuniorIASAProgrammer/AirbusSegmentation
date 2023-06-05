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
usage: train.py [-h] [--image-path] [--label-path] [--epochs]
                [--batch-size] [--learning-rate] [--save-model-path] [--augmentation] [--aug-batch-size] [--aug-iterations]

Train the U-Net on images

options:
  -h, --help            show this help message
  --image-path          Folder with train images
  --label-path          Path to .csv file
  --epochs              Number of epochs
  --batch-size          Batch size
  --learning-rate       Learning rate
  --save-model-folder   Folder name to save fitted model
  --augmentation        Apply data augmentation with ImageDataGenerator (True/False)
  --aug-batch-size      Batch of ImageDataGenerator (if --augmentation=True)
  --aug-iterations      Number of generated augmented batches from the batch ImageDataGenerator was fitted on (if --augmentation=True)
```

## Evaluate

```console
> python evaluate.py 
usage: evaluate.py [-h] [--image-path] [--prediction-path] [--batch-size]
                   [--model-path] [--threshold]

Predict masks with U-Net on images

options:
  -h, --help            show this help message
  --image-path          Folder with train images
  --prediction-path     Path to store predicted masks
  --batch-size          Batch size
  --model-path          Path to keras model
  --threshold           Threshhold for pixel value
```

## Conclusions
To solve this Segmentation task u-net CNN model was implemented. Labels were encoded in (start-run) format, so they needed to be preprocessed to 2D masks 0 or 1 pixels value (not-ship and ship). The dataset of images was pretty large (bigger than memory size of computing machine) so it was necessary to implement generators that scan images and pass them to model by batches. To make data more diverse, ImageDataGeterator was applied with some randoms shifts and flips to both original images and corresponding masks.  

Training process: GPU P100 on kaggle platform  

Metrics: dice and IOU

Results of test subset on pretrained model in **data/prediction/prediction.png**
