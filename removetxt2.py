import matplotlib.pyplot as plt
import keras_ocr
import math
import numpy as np
import cv2
import os

pipeline = keras_ocr.pipeline.Pipeline()

remove_list = ["September", "september", "1993"]

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

def inpaint_text(img, remove_list, pipeline):

    prediction_groups = pipeline.recognize([img])
    print(prediction_groups)
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        if box[0] in remove_list:
           x0, y0 = box[1][0]
           x1, y1 = box[1][1]
           x2, y2 = box[1][2]
           x3, y3 = box[1][3]

           x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
           x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

           print(x_mid0, y_mid0, x_mid1, y_mi1)

           thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))

           cv2.line(mask, (int(x_mid0 / 1.3), y_mid0), (int(x_mid1 * 1.3), y_mi1), 255,
           thickness)
           img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

    return(img)


image_paths = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg"]

# # Convert to grayscale image
# gray_img = cv2.cvtColor(image_paths, cv2.COLOR_BGR2GRAY)

# # Converting grey image to binary image by Thresholding
# thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

folder_path = "/Users/thomaskitson/Documents/newsguessr/1993"

for img_path in image_paths:
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    top_25 = img[0:int(height*0.25), 0:width]
    top_25 = inpaint_text(top_25, remove_list, pipeline)
    bottom_75 = img[int(height*0.25):height, 0:width]
    img = np.concatenate((top_25, bottom_75), axis=0)
    img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(folder_path, img_path), img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 10])
