import numpy as np
import cv2

def read_image_black_white(img_path):
    # Read img -> convert to black white
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    return img

def remove_empty_space(img, pre_crop=0):
    # img = cv2.imread('pg13_gau_preview.png') 
    if pre_crop > 0:
        img = img[:-pre_crop,:-pre_crop] # Perform pre-cropping
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    gray = 255*(gray < 128).astype(np.uint8) # To invert the text to white
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8)) # Perform noise filtering
    coords = cv2.findNonZero(gray) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    rect = img[y:y+h, x:x+w] # Crop the image - note we do this on the original image
    return rect

def padding_image(img, new_width, new_height,
            start_x=0, start_y=0, chanels=3,
            background=255):
    height, width = img.shape[:2]
    assert start_x + height <= new_height
    assert start_y + width <= new_width
    if chanels == 3:
        new_img = np.ones((new_height, new_width, 3), dtype=np.uint8) * background
        new_img[start_x: start_x + height, start_y: start_y + width, :] = img
    else:
        # grayscale
        new_img = np.ones((new_height, new_width), dtype=np.uint8) * background
        new_img[start_x: start_x + height, start_y: start_y + width] = img
    return new_img

def random_padding(img, max_width_height_ratio=20, min_width_height_ratio=10, chanels=3):
    height, width = img.shape[:2]
    max_width = int(height * max_width_height_ratio)
    if width >= max_width:
        return img
    min_width = int(height * min_width_height_ratio)
    min_width = max(min_width, width)
    new_width = np.random.randint(min_width, max_width)
    return padding_image(img, new_width, height, chanels=chanels)

