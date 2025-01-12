#!/usr/bin/env python

from PIL import Image
import torch
from facenet import Facenet
import os

image_dir = "img" 
image_names = []

model = Facenet()

# for filename in os.listdir(image_dir):
#   if filename.endswith(('.jpg', '.jpeg', '.png')):
#     image_names.append(filename)
image_names = sorted(
  [filename for filename in os.listdir(image_dir) if filename.endswith(('.jpg', '.jpeg', '.png'))]
)


for i in range(len(image_names)):
  for j in range(i + 1, len(image_names)):
    image_name1 = image_names[i]
    image_name2 = image_names[j]
    image1 = Image.open(os.path.join(image_dir, image_name1))
    image2 = Image.open(os.path.join(image_dir, image_name2))
    probability = model.detect_image(image1, image2)
    print(f'{image_name1} vs {image_name2}: {probability[0]}')