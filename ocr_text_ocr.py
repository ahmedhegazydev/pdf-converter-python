import cv2
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import pytesseract
from PIL import Image
import PIL.Image

import cv2
import numpy as np
import pytesseract
from PIL import Image
import wand

from pytesseract import image_to_string

# img = Image.open('E:\Python projects\FaceDetect-master - Copy\image.jpg')
img = Image.open('E:\Python projects\FaceDetect-master - Copy\img_aea.jpeg')
# text = image_to_string(img, lang="eng")
text = image_to_string(img, lang="ara")
print(text)
