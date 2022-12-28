# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import Image, display
import os
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import torch
import random
import time 
from PIL import Image
from glob import *
import skimage.io as io
import skimage.transform as trans
from tensorflow.keras.preprocessing import image
import shutil
import uuid
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np



## function read images Grad=========================================================================

def get_img_array(img_path, size):
    # `img` is a PIL image of size 64x64
    #img =image.load_img(img_path, target_size=(size,size,3))
    src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    resized=cv2.resize(src,(size,size))
    # `array` is a float32 Numpy array of shape (64,64, 3)
    img =image.img_to_array(resized)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 64, 64, 3)
    img = img/255
    img=img.reshape(1,size,size,3)
    return img

#=====================================================================================================

#GRAD_CAM ============================================================================================



def make_gradcam_heatmap(img_array, model, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.layers[10].output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


#save_gradcam===================================================================================
def save_gradcam(img_path, size,heatmap, alpha=0.4):
    # Load the original image
    #img = keras.preprocessing.image.load_img(img_path)
    src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    resized=cv2.resize(src,(size,size))
    # `array` is a float32 Numpy array of shape (64,64, 3)
    img =image.img_to_array(resized)
    #img = keras.preprocessing.image.img_to_array(img)
    #img = img/255
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Grad_Class_img the heatmap on original image
    Grad_Class_img = jet_heatmap * alpha + img

    return Grad_Class_img
#===================================================================================
##Functions to save images Unet===================================================================================

#Functions to save images Unet
#Step 4: Define function to save the test images
def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i] = color_dict[i]
      
    return img_out
#Function Clean Diroctory===============================================================================================

def delete_files(path_dir):
    for file_object in os.listdir(path_dir):
        file_object_path = os.path.join(path_dir , file_object)
        if os.path.isfile(file_object_path) or os.path.islink(file_object_path):
            os.unlink(file_object_path)
        else:
            shutil.rmtree(file_object_path)


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

#===============================================================================================    
#Function to make copies of Class image ================================================================

dir_dis="copies/Class/"

def copy_images(path_file,path_copy,num_image):
    for file_object in os.listdir(path_copy):
        file_object_path = os.path.join(path_copy , file_object)
        if os.path.isfile(file_object_path) or os.path.islink(file_object_path):
            os.unlink(file_object_path)
        else:
            shutil.rmtree(file_object_path)

    
    for pngfile in iglob(path_file):
        shutil.copy(pngfile,path_copy) 

    for item in os.listdir(path_copy):
        s = os.path.join(path_copy, item)
        if s.endswith(".png"): 
            files=sorted(os.listdir(path_copy))
            for item  in range(1,num_image):
               #filename = str(item ) + ".png"
    
               shutil.copy(s, os.path.join(path_copy, "test_" + str(item) + ".png"))  
    #return io.imread(os.path.join(path_copy, files[-1]))
#=======================================================================================================

##Unet Function to predict Single segmental image===================================================================================
#Unet Function to predict Single segmental image

#sav ve images mode =================================================

#Step 4: Define function to save the test images
def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i] = color_dict[i]
      
    return img_out


def saveResult(img_path,save_path,npyfile,flag_multi_class = False,num_class = 2):
    files=os.listdir(img_path)
    #print(len(img_path))
    #print(len(npyfile))
    
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        #img1=np.array(((img - np.min(img))/np.ptp(img))>0.6).astype(float)
        img[img>0.1]=1
        img[img<=0.1]=0
        io.imsave(os.path.join(save_path, files[i]+'_predict.png'),img)




#+=========================================================================

def testGenerator(test_path,num_image = 30,target_size = (512,512),flag_multi_class = False,as_gray = True):
    files=sorted(os.listdir(test_path))
    num_image=len(files)
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,files[i]),as_gray = as_gray)
        print(files[i])
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


#===========================================================================
def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.


#==========================================================================

#========Seeding the randomness
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

#Create a directory=========================================================
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
#Layer_augmentation_real_image==============================================
# data_augmentation =keras.Sequential(
#     [
#         layers.Normalization(),
#         layers.Resizing(64, 64),

#         layers.RandomFlip("horizontal_and_vertical"),
#         layers.RandomRotation(factor=0.02),
#         layers.RandomZoom(height_factor=0.2, width_factor=0.2),
#         layers.Rescaling(1./255)
#     ],
#     name="data_augmentation",
# )

#=================================================================================

#AUGMENTATION_REAL_IMAGES-DATA=================================================
datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=0.2, # rotation
        width_shift_range=0.2, # horizontal shift
        height_shift_range=0.2, # vertical shift
        zoom_range=0.2, # zoom
        horizontal_flip=True, # horizontal flip
        brightness_range=[0.2,1.2])  


#================================================================================