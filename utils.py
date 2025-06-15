import tensorflow as tf
from PIL import Image
import numpy as np
import os

def load_and_process(path, max_dim=512):
    img = Image.open(path)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.vgg19.preprocess_input(img[np.newaxis,...])
    return img

def deprocess(img):
    x = img.copy()
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[..., ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x[0]

def save_image(img_array, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img_array).save(path)