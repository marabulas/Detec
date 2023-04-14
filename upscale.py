import threading

import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)


def upscale(image, name):
    image = image

    sr_image = model.predict(image)

    sr_image.save(name + '.jpg')


# Code to upscale the image goes here
pass


def upscale_in_background(image, name):

    # Create a new thread and run the upscale function in it
    thread = threading.Thread(target=upscale, args=(image, name,))
    thread.start()

    # The main thread can continue to run other code while the upscale function is running in the background
    print("Upscaling in progress...")