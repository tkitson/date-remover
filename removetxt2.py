import matplotlib.pyplot as plt
import keras_ocr
import math
import numpy as np
import cv2

pipeline = keras_ocr.pipeline.Pipeline()

remove_list = ["1936"]

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)


def inpaint_text(img_path, remove_list, pipeline):


    img = keras_ocr.tools.read(img_path)
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        if box[0] in remove_list:
           x0, y0 = box[1][0]
           x1, y1 = box[1][1]
           x2, y2 = box[1][2]
           x3, y3 = box[1][3]

           x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
           x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

           thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))

           cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,
           thickness)
           img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

    return(img)

# img = inpaint_text(img_path, remove_list, pipeline)

# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# cv2.imwrite("text_free_image.jpg", img_rgb)

image_paths = ["2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg"]

for img_path in image_paths:
    img = inpaint_text(img_path, remove_list, pipeline)
    dim = (int(img.shape[1]*0.65),int(img.shape[0]*0.65))
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_path, img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 20])
