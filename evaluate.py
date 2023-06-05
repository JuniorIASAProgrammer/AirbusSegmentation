import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

from func import *
from tensorflow.keras.models import load_model

def test_generator(image_folder_path, batch_size):              
    """
    Yields the next data batch.
    """
    images = os.listdir(image_folder_path)
    for offset in range(0, len(images), batch_size):
        X_test = []
        image_names = []
        batch_images = images[offset:offset+batch_size]    
        for image_name in batch_images:
            image = cv2.imread(f"{image_folder_path}/{image_name}")
            image = cv2.resize(image, (256, 256), interpolation=None)
            X_test.append(image)
            image_names.append(image_name)
        X_test = np.array(X_test)/255
        yield X_test, image_names

# def plotImageWithMask(images, masks, binary=True):
#     total_images = len(images)
#     plot_height = total_images//3+1
    
#     if binary:
#         masks = (masks > 0.5).astype(np.uint8)
#     else:
#         masks = masks.astype(np.float32)
#     fig = plt.figure(figsize=(15, 15))
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

#     # get_pics
#     i = 0
#     for image, mask in zip(images, masks):
#         ax = fig.add_subplot(plot_height, 6, i + 1, xticks=[], yticks=[])
#         ax.imshow(cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2RGB))
#         ax = fig.add_subplot(plot_height, 6, i + 2, xticks=[], yticks=[])
#         ax.imshow(mask)
#         i+=2
#     fig.savefig('data/prediction/prediction.png')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--image-path', type=str, default="data/train_v2", help='Path to folder with train images', dest='image_path')
    parser.add_argument('--prediction-path', type=str, default="data/prediction", help='Path to store predicted masks', dest='prediction_path')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--model-path', dest='model_path', type=str, default="models/unet", help='Path to folder with keras model')
    parser.add_argument('--threshold', type=float, default=0.25, help='Threshhold for pixel binary classification (only if binary-mask is True)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    test_generator_instance = test_generator(args.image_path, args.batch_size)
    model = load_model(args.model_path, custom_objects={'dice_coef':dice_coef, 'iou_coef': iou_coef})
    model.summary()
    while True:
        try:
            X_test, image_names = next(test_generator_instance)
            y_pred = model.predict(X_test)
            masks = (y_pred > args.threshold).astype(np.uint8)*255
            for mask, name in zip(masks, image_names):
                cv2.imwrite(f'{args.prediction_path}/{name[:-4]}_mask.png', mask)
        except StopIteration:
            break




