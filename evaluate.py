import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import segmentation_models as sm
from func import *
from tensorflow.keras.models import load_model
from keras.metrics import MeanIoU

def test_generator(image_folder_path, batch_size, preprocessing='baseline'):              
    """
    Yields the next data batch.
    """
    images = os.listdir(image_folder_path)
    for offset in range(0, batch_size, batch_size):
        X_test = []
        image_names = []
        batch_images = images[offset:offset+batch_size]    
        for image_name in batch_images:
            image = cv2.imread(f"{image_folder_path}/{image_name}")
            image = cv2.resize(image, (256, 256), interpolation=None)
            X_test.append(image)
            image_names.append(image_name)
        X_test = np.array(X_test)
        if preprocessing != 'baseline':
            preprocess = sm.get_preprocessing(preprocessing)
            X_test = preprocess(X_test)
        yield X_test, image_names

def plotImageWithMask(images, masks, binary=True):
    total_images = len(images)
    plot_height = total_images//3+1
    
    if binary:
        masks = (masks > 0.5).astype(np.uint8)
    else:
        masks = masks.astype(np.float32)
    fig = plt.figure(figsize=(15, 15))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # get_pics
    i = 0
    for image, mask in zip(images, masks):
        ax = fig.add_subplot(plot_height, 6, i + 1, xticks=[], yticks=[])
        ax.imshow(cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2RGB))
        ax = fig.add_subplot(plot_height, 6, i + 2, xticks=[], yticks=[])
        ax.imshow(mask)
        i+=2
    fig.savefig('data/prediction/prediction.png')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--image-path', type=str, default="data/test_v2", help='Path to folder with train images', dest='image_path')
    parser.add_argument('--prediction-folder-name', type=str, default="pred_1", help='Path to store predicted masks: data/prediction/...', dest='prediction_path')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=16, help='Batch size (to use memory in efficient way)')
    parser.add_argument('--model-type', dest='model_type', type=str, default='baseline', help='Type of architecture: [baseline, vgg19, resnet34, efficientnetb5]')
    parser.add_argument('--model-path', dest='model_path', type=str, default="models/baseline", help='Path to folder with model')
    parser.add_argument('--threshold', type=float, default=0.25, help='Threshhold for pixel binary classification')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.mkdir(f'data/prediction/{args.prediction_path}')
    test_generator_instance = test_generator(args.image_path, args.batch_size, args.model_type)
    model = load_model(args.model_path, compile=False)

    model.summary()
    while True:
        try:
            X_test, image_names = next(test_generator_instance)
            y_pred = model.predict(X_test)
            masks = (y_pred > args.threshold).astype(np.uint8)*255
            for mask, name in zip(masks, image_names):
                cv2.imwrite(f'data/prediction/{args.prediction_path}/{name[:-4]}_mask.png', mask)
        except StopIteration:
            break




