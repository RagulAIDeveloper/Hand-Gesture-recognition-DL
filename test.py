from keras import utils
from keras.models import load_model
from keras.models import model_from_json
import numpy

json_file =open ("C:/Users/msrag/OneDrive/Desktop/Deep Learning/hand_prediction_03/model.json","r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("C:/Users/msrag/OneDrive/Desktop/Deep Learning/hand_prediction_03/model.h5")

def classify(img_file):
    img_name = img_file
    test_image = utils.load_img(img_name,target_size=(256,256),grayscale =True)
    test_image = utils.img_to_array(test_image)
    test_image = numpy.expand_dims(test_image,axis= 0)
    result = model.predict(test_image)
    arr = numpy.array(result[0])
    print("ARRAY",arr)
    maxx = numpy.amax(arr)
    max_prob = arr.argmax(axis=0)
    max_prob = max_prob+1
    print('max',max_prob)
    classes =["NONE","ONE","TWO","THREE","FOUR","FIVE"]
    result = classes[max_prob - 1]
    print("Img_name",img_name,"Result",result)

import os
path ="C:/Users/msrag/OneDrive/Desktop/Deep Learning/hand_prediction_03/test"

files =[]

for r,d,f in os.walk(path):
    for file in f:
        if ".jpg" in file:
            files.append(os.path.join(r,file))
for f in files:
    classify(f)
