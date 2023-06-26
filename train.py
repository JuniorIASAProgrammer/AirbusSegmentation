import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from func import *
from tensorflow.keras.optimizers import Adam
import argparse
import segmentation_models as sm

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--image-path', type=str, default="data/train_v2", help='Folder with train images', dest='image_path')
    parser.add_argument('--label-path', type=str, default="data/train_ship_segmentations_v2.csv", help='Path to .csv file', dest='label_path')
    parser.add_argument('--use-empty', type=str, default=False, help='Either use images with no ship objects or not', dest='use_empty')
    parser.add_argument('--model-type', type=str, default='baseline', dest='model_type', help='Type of architecture: [baseline, vgg19, resnet34, efficientnetb5]')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--augmentation', type=str, default=True, help='Either apply data augmentation with ImageDataGenerator or not')
    parser.add_argument('--aug-batch-size', dest='aug_batch_size', type=int, default=16, help='Batch of ImageDataGenerator')
    parser.add_argument('--aug-iterations', dest='aug_iterations', type=int, default=4, help='Number of generated augmented batches from the batch ImageDataGenerator was fitted')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate', dest='learning_rate')
    parser.add_argument('--save-model-folder', dest='save_model_folder', type=str, default="model_1", help='Folder name to save fitted model; path - models/{--save-model-folder}')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    # Create U-net
    optimizer = Adam(learning_rate=args.learning_rate)
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 0.5]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1*focal_loss)
    metrics = [dice_coef, sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    if args.model_type == 'baseline' :
        model = baseline_unet()
        preprocess = None
    else:
        model = sm.Unet(args.model_type, encoder_weights='imagenet', classes=1, activation='sigmoid')
        preprocess = args.model_type
    model.compile(optimizer, total_loss, metrics)
    model.summary()
    train_generator = data_generator(csv_data_file=args.label_path, 
                                     image_folder_path=args.image_path, 
                                     augmentation=args.augmentation,
                                     aug_batch_size=args.aug_batch_size,
                                     aug_cycles=args.aug_iterations,
                                     use_empty_images=args.use_empty,
                                     preprocessing=preprocess,
                                     batch_size=args.batch_size, 
                                     epochs=args.epochs, 
                                     )
    model_history = model.fit(train_generator, verbose=1)          
    model.save(f"models/{args.save_model_folder}")
    print(f"Fit loss - {model_history.history['loss']}")
    print(f"Fit dice - {model_history.history['dice_coef']}")
    print(f"Fit IOU - {model_history.history['iou_score']}")
    print(f"Fit F1 - {model_history.history['f1-score']}")