from func import *

from tensorflow.keras.optimizers import Adam
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--image-path', type=str, default="data/train_v2", help='Folder with train images', dest='image_path')
    parser.add_argument('--label-path', type=str, default="data/train_grouped_selected_ships.csv", help='Path to .csv file', dest='label_path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate', dest='learning_rate')
    parser.add_argument('--save-model-folder', dest='save_model_path', type=str, default="unet_v1", help='Folder name to save fitted model in models/{--save-model-folder} directory')
    parser.add_argument('--augmentation', type=str, default=False, help='Apply data augmentation with ImageDataGenerator')
    parser.add_argument('--aug-batch-size', dest='aug_batch_size', type=int, default=16, help='Batch of ImageDataGenerator')
    parser.add_argument('--aug-iterations', dest='aug_iterations', type=int, default=2, help='Number of generated augmented batches from the batch ImageDataGenerator was fitted')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    TOTAL_IMAGES = len(pd.read_csv(args.label_path))
    train_generator = data_generator(csv_data_file=args.label_path, 
                                     image_folder_path=args.image_path, 
                                     augmentation=args.augmentation,
                                     batch_size=args.batch_size, 
                                     epochs=args.epochs, 
                                     aug_batch_size=args.aug_batch_size,
                                     aug_iterations=args.aug_iterations)
    # Create U-net
    unet = unet_model()
    unet.compile(optimizer=Adam(learning_rate=args.learning_rate),
                loss='binary_crossentropy',
                metrics=[dice_coef, iou_coef])
    unet.summary()
    model_history = unet.fit(train_generator, verbose=1)          
    unet.save(f"models/{args.save_model_path}")
    print(f"Fit loss - {model_history.history['loss']}")
    print(f"Fit dice - {model_history.history['dice_coef']}")
    print(f"Fit IOU - {model_history.history['iou_coef']}")