import numpy as np
import pandas as pd
import cv2

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

def preprocess_labels(label_csv):
    ships = pd.read_csv(label_csv)
    ships['EncodedPixels'].replace(to_replace=np.nan, value="none", inplace=True)
    unique_ships = ships.groupby('ImageId', as_index=False)['EncodedPixels'].apply(' '.join)
    unique_ships_selected = unique_ships[unique_ships['EncodedPixels']!="none"]
    return unique_ships_selected

# convert EncodedPixels into 2d masks
def mask_converter(values):
    mask = np.zeros((768*768,), dtype=float)        #create empty one-dimentional vector with zeros
    if values != "none":
        values = values.strip().split()
        start_points = values[0::2]               #separate values
        lengths = values[1::2]
        for st_p, l in zip(start_points, lengths):     #fill mask with ones according to the EncodedPixels colomn
            st_p, l = int(st_p)-1, int(l)
            ones = np.ones(l, dtype=int) 
            mask[int(st_p):int(st_p)+int(l)] = ones
    return np.transpose(mask.reshape((768, 768, 1)), axes=(1, 0, 2))    #rotate image to get correct orientation

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,])
    union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def iou_coef(y_true, y_pred, smooth=1):
    true = K.flatten(y_true)
    pred = K.flatten(y_pred)
    intersection = K.sum(true*pred)
    union = K.sum(true) + K.sum(pred) - intersection
    return (intersection+smooth)/(union+smooth)

def iou_loss(y_true, y_pred):
    return -iou_coef(y_true, y_pred)

def data_generator(csv_data_file, image_folder_path, batch_size, epochs, augmentation=False, aug_batch_size=32):              
    """
    Yields the next data batch.
    """
    labels_file = pd.read_csv(csv_data_file)
    num_images = len(labels_file)
    for i in range(epochs):
        for offset in range(0, batch_size, batch_size):        
            # Get the samples you'll use in this batch
            batch_images = labels_file['ImageId'][offset:offset+batch_size].values
            batch_masks = labels_file['EncodedPixels'][offset:offset+batch_size].values
            # Initialise X_train and y_train arrays for this batch
            X_train = []
            y_train = []
            # For each example
            for image_filename, mask_encoded in zip(batch_images, batch_masks):          
                # Add example to arrays
                image = cv2.imread(f"{image_folder_path}/{image_filename}")
                mask = mask_converter(mask_encoded)
                image = cv2.resize(image, (256, 256), interpolation=None)
                mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                X_train.append(image)
                y_train.append(mask)

            X_train = np.array(X_train)/255
            y_train = np.array(y_train)[..., np.newaxis]

            if augmentation:
                image_gen = ImageDataGenerator(horizontal_flip=True,
                                               vertical_flip=True,
                                               width_shift_range=0.1,
                                               height_shift_range=0.1
                                                          )
                mask_gen = ImageDataGenerator(horizontal_flip=True,
                                              vertical_flip=True,
                                              width_shift_range=0.1,
                                              height_shift_range=0.1
                                                          )
                image_gen.fit(X_train)
                mask_gen.fit(y_train)
                counter = 0
                for X_aug_train, y_aug_train in zip(image_gen.flow(X_train, batch_size=aug_batch_size, seed=42), mask_gen.flow(y_train, batch_size=aug_batch_size, seed=42)):
                    if counter == 4:
                        counter = 0
                        break
                    counter += 1
                    yield X_aug_train, y_aug_train
            else:
                yield X_train, y_train
                # yield the next training batch

# initializing U-net model
def unet_model(input_size=(256,256,3)):
    inputs = Input(input_size)

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    d = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    return Model(inputs=[inputs], outputs=[d])