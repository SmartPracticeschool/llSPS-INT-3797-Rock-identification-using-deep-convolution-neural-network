# -*- coding: utf-8 -*-


from __future__ import division , print_function
import sys
import os 
import glob
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

from keras.models import load_model
from keras import backend
from tensorflow.keras import backend


import tensorflow as tf
global graph
graph = tf.get_default_graph()
from skimage.transform import resize
from flask import Flask,request,render_template,redirect,url_for
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
MODEL_PATH = 'trained_model.h5'
model = load_model(MODEL_PATH)

@app.route('/',methods = ["GET"] )
def index():
    return render_template("base.html")

@app.route('/predict',methods = ["GET","POST"])
def upload():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        print("current path: ",basepath)
        file_path = os.path.join(basepath,"uploads",secure_filename(f.filename))
        f.save(file_path)
        print("joined path: ",file_path)
        img = image.load_img(file_path,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis = 0)
       
        with graph.as_default():
            prediction_class = model.predict_classes(x)
            print("prediction",prediction_class)
        
        i=prediction_class.flatten()
        index = ['Basalt', 'Conglomerate', 'Dolostone', 'Gabbro', 'Gneiss', 'Granite', 'Limestone', 'Marble', 'Quartzite', 'Rhyolite', 'Sandstone', 'Shale', 'Slate']
        if(str(index[i[0]])=="Basalt"):
            text="This is " + str(index[i[0]] + " and it is an igneous rock.")
        elif (str(index[i[0]])=="Conglomerate"):
            text ="This is " + str(index[i[0]] + " and it is a sedimentary rock." )
        elif(str(index[i[0]])=="Dolostone"):
            text="This is " + str(index[i[0]] + " and it is a sedimentary rock.")
        elif(str(index[i[0]])=="Gabbro"):
            text="This is " + str(index[i[0]] + " and it is an igneous rock.")
        elif(str(index[i[0]])=="Gneiss"):
            text="This is " + str(index[i[0]] + " and it is a metamorphic rock.")
        elif(str(index[i[0]])=="Granite"):
            text="This is " + str(index[i[0]] + " and it is an igneous rock." )
        elif(str(index[i[0]])=="Limestone"):
            text="This is " + str(index[i[0]] +" and it is a sedimentary rock.")
        elif (str(index[i[0]])=="Marble"):
            text ="This is " + str(index[i[0]] +" and it is a metamorphic rock.")
        elif(str(index[i[0]])=="Quartzite"):
            text="This is " + str(index[i[0]] +" and it is a metamorphic rock.")
        elif(str(index[i[0]])=="Rhyolite"):
            text="This is " + str(index[i[0]] +" and it is an igneous rock.")
        elif(str(index[i[0]])=="Sandstone"):
            text="This is " + str(index[i[0]] +" and it is a sedimentary rock.")
        elif(str(index[i[0]])=="Shale"):
            text="This is " + str(index[i[0]] +" and it is a sedimentary rock.")
        elif(str(index[i[0]])=="Slate"):
            text="This is " + str(index[i[0]] +" and it is a metamorphic rock.")
    return text 
if __name__ == "__main__":
    app.run(debug = False ,threaded= False)
