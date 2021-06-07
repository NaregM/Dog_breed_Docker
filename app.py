import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2

#from glob import glob
#from tqdm import tqdm

import wikipedia

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os

import streamlit as st

from PIL import Image, ImageFile

#sns.set_style("darkgrid")
#sns_p = sns.color_palette('Paired')

# ==============================================
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

from PIL import Image
import requests
from io import BytesIO

# ==============================================

label = ""

st.markdown("<h1 style='text-align: center; border-color: red; color: dodgerblue;'>Dog Breed Identifier</h1>", unsafe_allow_html = True)

response = requests.get("https://streamlit-data.s3-us-west-1.amazonaws.com/all_dog.jpeg")
img = Image.open(BytesIO(response.content))

st.image(img, use_column_width = True)

# ===============================================

"### Given an image of a dog, this app will use Machine learning/AI methods to identify the dog’s breed. If a human image is provided, the algorithm will identify the dog breed that the human most resembles!!!"

"This project uses Convolutional Neural Networks and transfer learning in PyTorch environment."


"#### Start by uploading an image below: "


# ==============================================

model_transfer = models.resnet152(pretrained = True)

use_cuda = False

if use_cuda:

    model_transfer = model_transfer.cuda()


model_transfer = models.resnet152(pretrained = True)

use_cuda = False

if use_cuda:

    model_transfer = model_transfer.cuda()

# Freeze model weights
for param in model_transfer.parameters():

    param.requires_grad = False



new_layer = nn.Linear(model_transfer.fc.in_features, 133)
model_transfer.fc = new_layer


#import zipfile

#zip = zipfile.ZipFile('model_transfer.zip')
#zip.extractall()



model_transfer.load_state_dict(torch.load('model_transfer.pt',  map_location=torch.device('cpu')))



class_names = pd.read_csv('class_names.csv')
class_names = class_names['0'].values.tolist()


### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

# list of class names by index, i.e. a name can be accessed like class_names[0]
#class_names = [item[4:].replace("_", " ") for item in image_datasets['train'].classes]

def predict_breed_transfer(img_path):

    # load the image and return the predicted breed

    transform = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                         std = [0.229, 0.224, 0.225])
                                    ])

    img = Image.open(img_path)

    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    if use_cuda:

        batch_t = batch_t.cuda()

    model_transfer.eval()

    out = model_transfer(batch_t)

    _, index = torch.max(out, 1)

    if use_cuda:

        return class_names[np.squeeze(index.cpu().numpy())]

    else:

        return class_names[np.squeeze(index.numpy())]



def dog_detector(img_path):

    """
    Returns "True" if a dog is detected in the image stored at img_path
    """

    idx = VGG16_predict(img_path)

    if idx >= 151 and idx <= 268:

        return True

    else:

        return False

VGG16 = models.vgg16(pretrained = True)


def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to
    predicted ImageNet class for image at specified path

    Args:
        img_path: path to an image

    Returns:
        Index corresponding to VGG-16 model's prediction
    '''

    transform = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                         std = [0.229, 0.224, 0.225])
                                    ])

    img = Image.open(img_path)

    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    if use_cuda:

        batch_t = batch_t.cuda()

    VGG16.eval()

    out = VGG16(batch_t)

    _, index = torch.max(out, 1)

    if use_cuda:

        return np.squeeze(index.cpu().numpy())

    else:

        return np.squeeze(index.numpy())




def run_app(img_path):
    ## handle cases for a human face, dog, and neither

    dog = dog_detector(img_path)

    if dog:

        #plt.figure(figsize = (7, 6))

        #plt.imshow(plt.imread(img_path))

        return predict_breed_transfer(img_path)

    elif not dog:

        #plt.figure(figsize = (7, 6))

        #plt.imshow(plt.imread(img_path))

        res = predict_breed_transfer(img_path)

        return res


#st.title("Upload + Classification Example")

uploaded_file = st.file_uploader("", type = ["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption = 'Uploaded Image.', width = 512)#use_column_width = False)
    st.write("")
    st.write("Identifying...")

    label = run_app(uploaded_file)

    dog = dog_detector(uploaded_file)

    if dog:

        if label[0].lower() in ['a', 'o', 'i']:

            s1 = "<h1 style='text-align: center; color: #ff8c00;'> >> Your dog is a " +  str(label.lower()) + " <<" +  " </h1>"
            st.markdown(s1, unsafe_allow_html = True)

            try:
                "### Here is some cool information about your dog:\n", wikipedia.summary(label)
                st.markdown('Source: __Wikipedia__')

            except:

                "### No Wikipedia entry..."

        else:

            #'## Your dog is a ', label,'!'
            s1 = "<h1 style='text-align: center; color: #ff8c00;'> >> Your dog is a " +  str(label.lower()) + " <<" +  " </h1>"
            st.markdown(s1, unsafe_allow_html = True)

            try:
                "### Here is some cool information about your dog:\n", wikipedia.summary(label)
                st.markdown('Source: __Wikipedia__')

            except:

                "### No Wikipedia entry..."


    if not dog:

        if label[0].lower() in ['a', 'o', 'i']:

            s1 = "<h1 style='text-align: center; color: #ff8c00;'> >> You look like an " +  str(label.lower()) + " <<" +  " </h1>"
            st.markdown(s1, unsafe_allow_html = True)

            try:
                "### Here is some cool information about your dog:\n", wikipedia.summary(label)
                st.markdown('Source: __Wikipedia__')

            except:

                "### No Wikipedia entry..."

        else:

            #st.markdown("<h1 style='text-align: center; color: red;'>You look like </h1>", unsafe_allow_html = True, params = [label])
            s1 = "<h1 style='text-align: center; color: #ff8c00;'> >> You look like a " +  str(label.lower()) + " <<" +  " </h1>"
            st.markdown(s1, unsafe_allow_html = True)

            try:
                "### Here is some cool information about your dog:\n", wikipedia.summary(label)
                st.markdown('Source: __Wikipedia__')

            except:

                "### No Wikipedia entry..."


# Search more dog pics and show them
if len(label) != 0:
    "### See some", label, " images from social media below:"
    st.markdown("Under construction")


st.markdown('-------------------------------------------------------------------------------')

st.markdown("""
                Made by [Nareg Mirzatuny](https://github.com/NaregM)

Source code: [GitHub](
                https://github.com/NaregM/Dog_breed_Docker)

""")
st.markdown('-------------------------------------------------------------------------------')
