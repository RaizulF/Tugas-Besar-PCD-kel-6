import cv2
import numpy as np
import os
import random
from concurrent.futures import ThreadPoolExecutor

class ImageAugmentor:
    def __init__(self, data, labels, file_names, output_path):
        self.data = data
        self.labels = labels
        self.file_names = file_names
        self.output_path = output_path
        self.augmentations = [self.flip, self.rotate, self.scale, self.translate]

    def flip(self, image):
        # Flip the image horizontally or vertically
        flip_code = random.choice([-1, 0, 1])  # -1: both axes, 0: vertical, 1: horizontal
        return cv2.flip(image, flip_code)

    def rotate(self, image):
        # Rotate the image by a random angle
        angle = random.uniform(-30, 30)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        return cv2.warpAffine(image, M, (w, h))

    def scale(self, image):
        # Scale the image by a random factor
        scale_factor = random.uniform(0.5, 1.5)
        h, w = image.shape[:2]
        return cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))

    def translate(self, image):
        # Translate the image by a random offset
        h, w = image.shape[:2]
        tx = random.uniform(-0.2 * w, 0.2 * w)
        ty = random.uniform(-0.2 * h, 0.2 * h)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, M, (w, h))

    def random_augment(self, image):
        # Apply a random augmentation
        augmentation = random.choice(self.augmentations)
        return augmentation(image)

    def process_image(self, image, label, file_name):
        augmented_image = self.random_augment(image)
        return augmented_image, label, file_name

    def augment_data(self):
        augmented_images = []
        for image, label, file_name in zip(self.data, self.labels, self.file_names):
            augmented_image, _, _ = self.process_image(image, label, file_name)
            augmented_images.append((augmented_image, label, file_name))
        self.save_augmented_images(augmented_images)

    def save_augmented_images(self, augmented_images):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        for augmented_image, label, file_name in augmented_images:
            label_folder = os.path.join(self.output_path, label)
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)
            output_path = os.path.join(label_folder, f'{file_name[:-4]}_augmented.jpg')
            cv2.imwrite(output_path, augmented_image)

        # Save original images
        for image, label, file_name in zip(self.data, self.labels, self.file_names):
            label_folder = os.path.join(self.output_path, label)
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)
            output_path = os.path.join(label_folder, f'{file_name[:-4]}.jpg')
            cv2.imwrite(output_path, image)
    

