import streamlit as st
import numpy as np
from tensorflow import keras
import tensorflow as tf
from PIL import Image, ImageOps
#from tensorflow.keras import backend as K

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

def loading_model():
  import zipfile
  with zipfile.ZipFile("pruned_res50_model.zip", 'r') as zip_ref:
    zip_ref.extractall()

  fp = "pruned_vgg_model.h5"
  model_loader = keras.models.load_model(fp)
  return model_loader

def import_predict(image_data,model):
  size = (224,224)
  image = ImageOps.fit(image_data,size)
  image = np.asarray(image)
  resized_image = image.reshape(1,224,224,3)
  processed_image = keras.applications.vgg16.preprocess_input(resized_image)
  prediction = model.predict(processed_image)
  conf = np.max(prediction)
  return prediction, conf

model = loading_model()
st.write("""
# X-Ray Image Classification (Pneumonia/Normal)  by Y.Q.
""")

temp_file = st.file_uploader("Upload X-Ray Image (Note: image's height and width must be larger than 224 pixels)")

if(temp_file is None):
  st.text("Please upload a X-Ray image file here ")

else:
  image = (Image.open(temp_file)).convert('RGB')
  pred,conf = import_predict(image,model)

  if pred[0][1] > 0.5:
    out = ('This is a Pneumonia case')
    st.success(out)
  else:
    out = ('This is a Normal case')
    st.success(out)

  st.success("Confidence Level (0 to 1): " + str(conf))
  st.image(image, use_column_width=True)

st.text("github link: https://github.com/qiuyichen00/Pneumonia-Classifier")
