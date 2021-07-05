import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras import backend as K

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)


def loading_model():
  import zipfile
  with zipfile.ZipFile("pruned_vgg_model.zip", 'r') as zip_ref:
    zip_ref.extractall()

  fp = "pruned_vgg_model.h5"
  model_loader = load_model(fp)
  return model_loader

def import_predict(image_data,model):
  size = (224,224)
  image = ImageOps.fit(image_data,size)
  image = np.asarray(image)
  resized_image = image.reshape(1,224,224,3)
  processed_image = tf.keras.applications.vgg16.preprocess_input(resized_image)
  prediction = model.predict(processed_image)
  conf = np.max(prediction)
  st.success(str(conf))
  return prediction

model = loading_model()
st.write("""
# X-Ray Classification (Pneumonia/Normal) -by Y.Q.
""")

temp_file = st.file_uploader("Upload X-Ray Image")

if(temp_file is None):
  st.text("Please upload a X-Ray image file here")

else:
  image = (Image.open(temp_file)).convert('RGB')
  st.image(image,use_column_width=True)
  pred = import_predict(image,model)

  out = (str(pred[0][1]))

  if pred[0][1]== 1:
    out = ('this is a Pneumonia case')
    st.success(out)
  else:
    out = ('this is a Normal case')
    st.success(out)
