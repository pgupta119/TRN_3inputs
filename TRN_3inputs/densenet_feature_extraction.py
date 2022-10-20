import tensorflow
import keras
from tensorflow.keras.applications import DenseNet121
from  tensorflow.keras.applications.densenet import  preprocess_input
model = DenseNet121(weights='imagenet',include_top= False)
# penultimate_layer = model.layers[-2]
# new_top_layer = keras.layers.Dense(112)(penultimate_layer.output) # create new FC layer and connect it to the rest of the model
# new_new_top_layer = keras.layers.AveragePooling2D(pool_size=(1,1), strides=None, padding='valid', data_format=None)(new_top_layer)
# new_model = keras.models.Model(inputs=model.input, outputs=new_new_top_layer)
# model.summary()

import os
a= []
for folder in sorted(os.listdir(f'/workspace/persistent/TRN.pytorch/data/HDD/camera')):
   a.append(folder)

from tqdm import tqdm
from tensorflow.keras.utils import load_img, img_to_array
# from google.colab import files
import os
import numpy as np
from keras.preprocessing import image
# import tqdm
import glob
# file  =/content/drive/MyDrive

# for  i in
for i in tqdm(range(64,len(a))):
  num_array = []
  paths = "/workspace/persistent/TRN.pytorch/data/HDD/camera"
  paths = os.path.join(paths, a[i] + '/*.jpg')
  print(a[i])
  for file in sorted(glob.glob(paths)):
    print(file)
    # a= cv2.imread(file)
    img = tensorflow.keras.utils.load_img(file, target_size=(320,320))
    img_data = tensorflow.keras.utils.img_to_array(img)
    # img_data= (img_data-128)/128
    # print(img_data.size)
    img_data = np.expand_dims(img_data, axis=0)
    # print(img_data.size)
    img_data = preprocess_input(img_data)
    # print(img_data.size)
    # model.add(AveragePooling2D())
    en1 = model.predict(img_data)
    # print(en1.size)
    # print(en1.size)
    # enl= en1.resize(98304,refcheck=False)
    en1 = en1.reshape(1600,8,8)
    num_array.append(en1)
  num_array= np.array(num_array)
  save_file ="/workspace/persistent/TRN.pytorch/data/HDD/densenet"
  save_file =os.path.join(save_file,a[i])
  np.save(save_file,num_array)
