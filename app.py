# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
from pathlib import Path, PureWindowsPath
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import sys
from PIL import Image
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten ,ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import os as let 


from tensorflow.keras.regularizers import l2
from tensorflow.keras import *
from glob import *
#import os
import uuid
import urllib.request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import *
from skimage import io
from tensorflow.keras.preprocessing import image
import cv2
from segment_model import *
import numpy as np
from functions import *
from data_utils import auto_body_crop

UPLOAD_FOLDER = './flask app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__, static_url_path='/assets',
            static_folder='./flask app/assets',
            template_folder='./flask app')
path_segmented ="segmented"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['path_segmented']= path_segmented
img_size=119


# Model saved with Keras model.save()

# You can also use pretrained model from Keras


#================================Load the Models that used for predicting  ==========================================
#load path the GMD_segment model that used for prdiction Original data
model_GMD=tf.keras.models.load_model('models/Model.h5' ,compile=False)
#load path the UneT_segment model that used for  segmenting Data
model_Unet=tf.keras.models.load_model('models/unet_STARE.hdf5',compile=True)
model_Unet.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = 'accuracy')


#load path the GMD_segment model that used for prdicting segmented Data
Model_SGMD=tf.keras.models.load_model('models/Model.h5',compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')

#====================================================================================================


@app.route('/')
def root():
    return render_template('index.html')


@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/upload.html')
def upload():
    return render_template('upload.html')


@app.route('/contact.html')
def contact():
    return render_template('contact.html')


@app.route('/news.html')
def news():
    return render_template('news.html')


@app.route('/reseaches.html')
def reseaches():
    return render_template('reseaches.html')


@app.route('/upload_img.html')
def upload_img():
    return render_template('uploaded_img.html')


@app.route('/upload_Simg.html')
def upload_Simg():
    return render_template('upload_Simg.html')
##==========================================================================================
             # here for fisrt integrations model ( GMD + Grad_Cam )
##==========================================================================================
@app.route('/uploaded_img', methods=['POST', 'GET'])
def uploaded_chest():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect('upload.html')
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect('upload.html')
        if file:
            filename = secure_filename(file.filename)
            file_path =os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(file_path)

            src = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            resized=cv2.resize(src,(img_size,img_size))
            #img_augmented= data_augmentation(src)
            #AUGMENTATION_REAL_IMAGES-DATA=================================================

            img = image.img_to_array(resized)
            img = img/255 

            #img_augmented = datagen.train_generator
            #resized=cv2.resize(test_generator,(img_size,img_size))




            #================================================================================
            # img = image.img_to_array(resized)
            # img = img/255
            #filena = Path(file_path)

            # Convert path to Windows format
            #path_on_windows = PureWindowsPath(filena)
            img_Class=get_img_array(file_path,119)
            heatmap = make_gradcam_heatmap(img_Class, model_GMD)
            
            Grad_Class =save_gradcam(file_path,199,heatmap)
            # Rescale heatmap to a range 0-255
            #Grad_Class=save_gradcam(file_path,heatmap)
            filename1 = my_random_string(10) + filename
            filename2 = my_random_string(10) + filename
            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename1))
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),Grad_Class)
            print("GradCAM image saved ")


            Classes=['DiabeticRetinopathy','Myopia', 'Glaucoma','Normal']
            pred_class = model_GMD.predict(img.reshape(1,119,119,3))
            print(pred_class)
            top_2 = np.argsort(pred_class[0])[:-4:-1]
            result=[]
            for i in range(2):
                   pred_proba="{}".format(Classes[top_2[i]])+" ({0:.2%})".format(pred_class[0][top_2[i]])
                   result.append(pred_proba)
            print(pred_proba)
            print(pred_class)
            # check if models exists
            return render_template('results_GMD.html', result=result,imagesource='/assets/images/'+filename1,imagesource1='/assets/images/'+filename2)

##==========================================================================================
##==========================================================================================



##==========================================================================================
             # here for second  integrations model ( GMDS + Unet segmentation )
##==========================================================================================

@app.route('/uploaded_Simg', methods=['POST', 'GET'])
def uploaded_ct():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect('upload.html')
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect('upload.html')
        if file:
            filename = secure_filename(file.filename)
            file_path =os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(file_path)
            copy_images(file_path,dir_dis,3)
            delete_files(path_segmented)

            #====this part of code used by tensorflow backend to generate singal segmental images ===========
            #================================================================================================
            #Step 1: Run model on test images and save the images
            #number of test images
            #n_i=len(os.listdir(dir_dis))
            #Call test generator
            #test_gen = testGenerator(dir_dis)
            #Return model outcome for each test image
            #results = model_Unet.predict_generator(test_gen,n_i,verbose=1)
            #saveResult(dir_dis,path_segmented,results)
            #===================================================================================================
            #===================================================================================================

            #============================= here we used Pytorch backend to generate new data mask image segment======
            #========================================================================================================
            """ Load dataset """
            images_list = sorted(glob("copies/Class/*"))
            segment_Unet(images_list,path_segmented)
            #========================================================================================================
            #========================================================================================================
            filename1 = my_random_string(10) + filename
            #file_w = Path(path_segmented)

            # Convert path to Windows format
            #path_on_window = PureWindowsPath(file_w)
            file_s=os.listdir(path_segmented)
            filename2 = my_random_string(10) + str(file_s[-1])
            path_s=os.path.join(path_segmented, str(file_s[-1]))
            
            
            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename1))
            src = cv2.imread(path_s, cv2.IMREAD_UNCHANGED)
            io.imsave(os.path.join(app.config['UPLOAD_FOLDER'],filename2),src)
            
            src = cv2.imread(path_s, cv2.IMREAD_UNCHANGED)
            resized=cv2.resize(src,(img_size,img_size))
            img = image.img_to_array(resized)
            img = img/255
            Classes=['DiabeticRetinopathy','Myopia', 'Glaucoma','Normal']
            pred_class = Model_SGMD.predict(img.reshape(1,119,119,3))
            top_1 = np.argsort(pred_class[0])[:-4:-1]
            result=[]
            for i in range(1):
                   pred_proba="{}".format(Classes[top_1[i]])+"({0:.2%})".format(pred_class[0][top_1[i]])
                   result.append(pred_proba)
            print(pred_proba)
            print(pred_class)
           
            return render_template('results_S_GMD.html',result=result,imagesource='/assets/images/'+filename1,imagesource1='/assets/images/'+filename2)

##==========================================================================================
##==========================================================================================        
if __name__ == '__main__':
    app.secret_key = ".."
    port = int(os.environ.get('PORT',5000))
    app.run(debug=True,host='127.0.0.1', port=port)
