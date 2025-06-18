## Visualist some of the images in each class

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# load
data_dir = "./data"

# get the list of classes
class_names = os.listdir(data_dir)

# print the list of classes
print(class_names)

# loop through the classes and print the number of images in each class
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    num_images = len(os.listdir(class_dir))
    print(f"{class_name}: {num_images}")

# Display some images with subplots
def display_images(pth):
    figure, axis = plt.subplots(2, 2)
    for i in range(2):
        for j in range(2):
            class_name = class_names[i * 2 + j]
            class_dir = os.path.join(pth, class_name)
            images = os.listdir(class_dir)
            for image in images:
                img_path = os.path.join(class_dir, image)
                img = cv2.imread(img_path)
                
                axis[i, j].imshow(img)
                axis[i, j].set_title(class_name)
                axis[i, j].axis("off")
    plt.show()

## process the iamges
new_path = "./processed_data"

#create new directory
os.makedirs(new_path, exist_ok=True)

def process_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

# loop through the classes and process the images

def rotate_img(img, deg):
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return img


for class_name in class_names:
    # create new directory
    os.makedirs(os.path.join(new_path, class_name), exist_ok=True)


for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    images = os.listdir(class_dir)
    for image in images:
        class_name = str(class_name)
        img_path = os.path.join(class_dir, image)

        # process the image
        img = process_image(img_path)

        # save the image
        cv2.imwrite(os.path.join(new_path, class_name, image), img)

        #rotate the image for all 3 other rotations
        for target_class in range(4):
            source_class = int(class_name)
            if target_class == source_class: continue
            
            # Calculate the required rotation based on the relationship between classes
            # (target_class - source_class) % 4 gives us how many 90° rotations we need
            # Multiply by 90 to get degrees
            rotation_steps = (target_class - source_class) % 4
            if rotation_steps == 1:
                rotated_img = rotate_img(img, 90)
            elif rotation_steps == 2:
                rotated_img = rotate_img(img, 180)
            elif rotation_steps == 3:
                rotated_img = rotate_img(img, 270)
            
            # Save the rotated image in the appropriate class folder
            cv2.imwrite(os.path.join(new_path, str(target_class), f"rotated_{source_class}to{target_class}_{image}"), rotated_img)


