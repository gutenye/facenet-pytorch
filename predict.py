#!/usr/bin/env python

from PIL import Image
import torch

from facenet import Facenet

if __name__ == "__main__":
    model = Facenet()
    image_1 = "img/40.jpg"
    image_2 = "img/41.jpg"
    image_1 = Image.open(image_1)
    image_2 = Image.open(image_2)
    probability = model.detect_image(image_1,image_2)
    print(probability)
