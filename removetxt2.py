import matplotlib.pyplot as plt
import keras_ocr
import math
import numpy as np
import cv2
import os

pipeline = keras_ocr.pipeline.Pipeline()


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

papers_folder = "/Users/thomaskitson/Documents/newsguessr/papers"
subfolders = [f.path for f in os.scandir(papers_folder) if f.is_dir()]

for folder in subfolders:
    new_folder = os.path.join(folder, "new")
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    image_paths = [f.path for f in os.scandir(folder) if f.is_file() and f.name.endswith('.jpg') and f.name !='.DS_Store']
    base_name = os.path.basename(folder)
    remove_list = base_name.split(" ")
    print(remove_list)
    for img_path in image_paths:
        print(img_path)
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        top_25 = img[0:int(height*0.25), 0:width]
        top_25 = inpaint_text(top_25, remove_list, pipeline)
        bottom_75 = img[int(height*0.25):height, 0:width]
        img = np.concatenate((top_25, bottom_75), axis=0)
        img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        new_path = os.path.join(new_folder, os.path.basename(img_path))
        cv2.imwrite(new_path, img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 10])
        # cv2.imwrite(os.path.join(folder, img_path), img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 10])
