import matplotlib.pyplot as plt
import keras_ocr
pipeline = keras_ocr.pipeline.Pipeline()
#read image from the an image path (a jpg/png file or an image url)
img = keras_ocr.tools.read("The_London_Evening_Standard_Thu__Jan_15__1987_.jpg")
# Prediction_groups is a list of (word, box) tuples
prediction_groups = pipeline.recognize([img])
#print image with annotation and boxes
# keras_ocr.tools.drawAnnotations(image=img, predictions=prediction_groups[0])

print(prediction_groups[0])
