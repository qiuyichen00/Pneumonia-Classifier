# Pneumonia-Classifier
This is a project aiming to get a model to classify if a patient has pneumonia.
With input of a X-Ray Image, user can get a result saying if the patient is pneumonia-positive or normal.
A Confidence Level from 0 to 1 is also Provided.

 The following is confusion metrics of performace on model, trained by transferred-learning of res50 architecture on imagenet dataset.
 ![image](https://user-images.githubusercontent.com/46574239/124470292-09105980-ddce-11eb-8b9d-6d901c229195.png)

The code to train the classifier is written in Python, mainly with Tensorflow's Keras module. 
To see the whole process of training the model, please visit the following Google Colab link:
https://colab.research.google.com/drive/1c31H_h0SEhxQrtv3YSEP2w7OcLTHaxD6?usp=sharing

The app in this GitHub Repository is deployed on heroku with the link below:
 https://pneumonia-classifier-app.herokuapp.com/
 


