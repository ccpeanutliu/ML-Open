import tensorflow as tf
import keras
import sys
import csv
import math
import numpy as np
from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten  ,Activation, Input, Convolution2D,GlobalAveragePooling2D,DepthwiseConv2D
from keras.optimizers import SGD,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
typecount=7
label=[]
data=[]
count=0
with open(sys.argv[1], newline='') as csvFile:
#with open('/Users/peter yang/Downloads/train.csv', newline='') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		if count==0:
			count+=1
			continue
		data.append([])
		label.append([])
		label[count-1].append(row[0])
		for j in range(1,len(row)):		
			data[count-1].append(row[j])
		count+=1
sdata=[]
for i in range(len(data)):
	sdata.append(data[i][0].split())

sdata=np.array(sdata,dtype=float)
sdata=np.reshape(sdata,(len(data),48,48))
augmantdata=[]
auglabel=[]
crosssize=[[0,0],[0,6],[6,0],[6,6],[3,3]]
count1=0
for i in range(len(sdata)):
	for j in range(5):
		augmantdata.append([])
		auglabel.append(label[i][0])
		for x in range(crosssize[j][0],42+crosssize[j][0]):
			for y in range(crosssize[j][1],42+crosssize[j][1]):
				augmantdata[count1].append(sdata[i][x][y])
		count1+=1
trainlabel=[]
for i in range(len(augmantdata)):
	trainlabel.append([])
	for j in range(typecount):
		if(j==int(auglabel[i])):
			trainlabel[i].append(1)
		else:
			trainlabel[i].append(0)
augmantdata=np.array(augmantdata,dtype=float)
augmantdata/=255.0
del sdata
del label
del auglabel
trainlabel=np.array(trainlabel,dtype=int)
augmantdata=np.reshape(augmantdata,(5*len(data),42,42,1))
vdata=[]
vlabel=[]
count=0
augmantdata-=np.mean(augmantdata)
augmantdata/=np.std(augmantdata)
alpha=1
drop=0.3
model = Sequential()
model.add(Convolution2D(int(32 ), (3, 3), strides=(2, 2), padding='same' ,activation='linear',input_shape=(42,42,1)))
model.add(LeakyReLU(alpha=0.01))
model.add(BatchNormalization())
model.add (DepthwiseConv2D( (3, 3), strides=1, padding='same' ,activation='linear'))
model.add (LeakyReLU(alpha=0.1))
model.add (BatchNormalization())
model.add(Convolution2D(int(32 ), (1, 1), strides=(1, 1), padding='same' ,activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add (DepthwiseConv2D( (3, 3), strides=(2, 2), padding='same' ,activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add (Convolution2D(int(32 ), (1, 1), strides=(1, 1), padding='same' ,activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add (DepthwiseConv2D( (3, 3), strides=(1, 1), padding='same' ,activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Convolution2D(int(64 ), (1, 1), strides=(1, 1), padding='same' ,activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add (DepthwiseConv2D( (3, 3), strides=(2, 2), padding='same' ,activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add (Convolution2D(int(96 ), (1, 1), strides=(1, 1), padding='same' ,activation='linear'))
model.add(LeakyReLU(alpha=0.01))
model.add(BatchNormalization())

model.add (DepthwiseConv2D( (3, 3), strides=(1, 1), padding='same' ,activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Convolution2D(int(128 ), (1, 1), strides=(1, 1), padding='same' ,activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add (DepthwiseConv2D( (3, 3), strides=(2, 2), padding='same' ,activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add (Convolution2D(int(128 ), (1, 1), strides=(1, 1), padding='same' ,activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
"""
model.add (DepthwiseConv2D( (3, 3), strides=(1, 1), padding='same' ,activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add (Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same' ,activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
"""

model.add(Dropout(drop))
model.add (GlobalAveragePooling2D())
"""
model.add(Flatten())


model.add(Dense(units=32,))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(drop))
"""
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam' ,metrics=['accuracy'])
model.summary()

gen = ImageDataGenerator( horizontal_flip=True ,
						validation_split=0.1)
gen.fit(augmantdata)
train_generator = gen.flow(augmantdata, trainlabel, batch_size=256,subset='training')
validation_generator = gen.flow(
    augmantdata, trainlabel,
    batch_size=256,
    subset='validation')
learning_rate_function = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2,
                                            min_delta=0.0001, 
                                            verbose=1, 
                                            factor=0.1)
model.fit_generator(train_generator, epochs=18,
                                    verbose=1,
                                    shuffle=True,
                                    steps_per_epoch=1000,
                                #   class_weight=cw,

                                    callbacks=[learning_rate_function],
                                    validation_data = validation_generator,
                                    validation_steps= len(augmantdata)//2560
                                    )

weight=model.get_weights()
np.savez_compressed('weight.npz', weight)
model.save_weights('my_weight.h5')
#	python HW3.py train.csv test1.csv