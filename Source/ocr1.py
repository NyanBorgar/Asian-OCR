import numpy as np
import pytesseract
#import argparse
import cv2
from PIL import ImageFont, ImageDraw, Image
import os

os.environ['OMP_THREAD_LIMIT'] = '4'
path=os.getcwd()
path = path+'\\Tesseract-OCR\\tesseract'
pytesseract.pytesseract.tesseract_cmd = path
path1=os.getcwd()
path1 = path1+'\\Tesseract-OCR\\tessdata'
class ocr:
    def actual(imgs):
        image = cv2.imread(imgs)
        orig = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        config = ("-l chi_sim+jpn+chi_tra --oem 0 --psm 11 --tessdata-dir 'Tesseract-OCR\\tessdata'")
        text = pytesseract.image_to_boxes(gray,config=config)

        font = ImageFont.truetype('Font\\MEIRYO.TTC',20)

        output = image.copy()
        h,w,c = output.shape
        img_pil = Image.fromarray(output)
        draw = ImageDraw.Draw(img_pil)
        for box in text.splitlines():
             box = box.split(' ')
             #output = cv2.rectangle(output, (int(box[1]),h - int(box[2])), (int(box[3]), h - int(box[4])),(0, 0, 255), 2)
             draw.rectangle([(int(box[1]),h - int(box[2])), (int(box[3]), h - int(box[4]))], outline = (0,0,255,0))
             draw.text((int(box[1]), h - int(box[2])), box[0], font=font, fill=(0,255,0,0))     
        output = np.array(img_pil)
        
        texts = pytesseract.image_to_string(gray, config=config)
        return(texts,image,output)
