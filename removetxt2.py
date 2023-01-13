import matplotlib.pyplot as plt
import keras_ocr
import math
import numpy as np
import cv2

pipeline = keras_ocr.pipeline.Pipeline()

remove_list = ["thursday", "15", "january", "1987"]

img_path = "The_London_Evening_Standard_Thu__Jan_15__1987_.jpg"

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

img = inpaint_text(img_path, remove_list, pipeline)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imwrite("text_free_image.jpg", img_rgb)


# # https://youtu.be/3RNPJbUHZKs
# """
# Remove text from images

# """

# import matplotlib.pyplot as plt
# import keras_ocr
# import cv2
# import math
# import numpy as np

# remove_list = ["not", "GUILTY"]

# #General Approach.....
# #Use keras OCR to detect text, define a mask around the text, and inpaint the
# #masked regions to remove the text.
# #To apply the mask we need to provide the coordinates of the starting and
# #the ending points of the line, and the thickness of the line

# #The start point will be the mid-point between the top-left corner and
# #the bottom-left corner of the box.
# #the end point will be the mid-point between the top-right corner and the bottom-right corner.
# #The following function does exactly that.
# def midpoint(x1, y1, x2, y2):
#     x_mid = int((x1 + x2)/2)
#     y_mid = int((y1 + y2)/2)
#     return (x_mid, y_mid)

# #Main function that detects text and inpaints.
# #Inputs are the image path and kreas_ocr pipeline
# def inpaint_text(img_path, remove_list, pipeline):
#     # read the image
#     img = keras_ocr.tools.read(img_path)

#     # Recogize text (and corresponding regions)
#     # Each list of predictions in prediction_groups is a list of
#     # (word, box) tuples.
#     prediction_groups = pipeline.recognize([img])

#     #Define the mask for inpainting
#     mask = np.zeros(img.shape[:2], dtype="uint8")
#     for box in prediction_groups[0]:
#         if box[0] in remove_list:

#             x0, y0 = box[1][0]
#             x1, y1 = box[1][1]
#             x2, y2 = box[1][2]
#             x3, y3 = box[1][3]

#             x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
#             x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

#             #For the line thickness, we will calculate the length of the line between
#             #the top-left corner and the bottom-left corner.
#             thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))

#             #Define the line and inpaint
#             cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,
#             thickness)
#             inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

#     return(inpainted_img)

# # keras-ocr will automatically download pretrained
# # weights for the detector and recognizer.
# pipeline = keras_ocr.pipeline.Pipeline()

# img_text_removed = inpaint_text('The_London_Evening_Standard_Thu__Jan_15__1987_.jpg', remove_list, pipeline)

# plt.imshow(img_text_removed)

# cv2.imwrite('text_removed_image.jpg', cv2.cvtColor(img_text_removed, cv2.COLOR_BGR2RGB))
