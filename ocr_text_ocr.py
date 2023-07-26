
import pytesseract
from PIL import Image



from pytesseract import image_to_string

# img = Image.open('E:\Python projects\FaceDetect-master - Copy\image.jpg')
img = Image.open(r'E:\database_files\2023\data\arabic-words\mokhtar\f6aa1063-0ae5-4f3d-8762-84019c65b21f-015.jpg')

#pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\youssri.ahmed\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# text = image_to_string(img, lang="eng")
text = image_to_string(img, lang="ara")

print(text)
