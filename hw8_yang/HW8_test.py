import tensorflow as tf
import keras
import sys
import csv
import math
import numpy as np
from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D  ,Activation
from keras.layers.normalization import BatchNormalization
from keras.initializers import he_normal
from keras.layers import Dense,Dropout,Flatten
from keras.optimizers import SGD,Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Convolution2D,GlobalAveragePooling2D,DepthwiseConv2D


alpha=1
drop=0.5
w=np.load('weight.npz')

print(w['arr_0'])
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
model.add(Dropout(drop))
model.add (GlobalAveragePooling2D())
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam' ,metrics=['accuracy'])
model.set_weights(w['arr_0'])
#model.load_weights('my_weight.h5')
mean=0.520694489754212
std=0.2449954906086243
typecount=7
label=[]
data=[]
count=0
with open(sys.argv[1], newline='') as csvFile:
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
trainlabel=[]

for i in range(len(data)):
	trainlabel.append([])
	for j in range(typecount):
		if(j==int(label[i][0])):
			trainlabel[i].append(1)
		else:
			trainlabel[i].append(0)
trainlabel=np.array(trainlabel,dtype=int)

cut=20
leng=len(data)/cut
leng=int(leng)
output=[]
crosssize=[[0,0],[0,6],[6,0],[6,6],[3,3]]
for c in range(cut-1):
	sdata=[]
	for i in range(c*leng,(c+1)*leng):
		sdata.append(data[i][0].split())
	sdata=np.array(sdata,dtype=float)
	sdata=np.reshape(sdata,(len(sdata),48,48))
	augmantdata=[]
	count1=0
	for i in range(len(sdata)):
		for j in range(5):
			augmantdata.append([])
			for x in range(crosssize[j][0],42+crosssize[j][0]):
				for y in range(crosssize[j][1],42+crosssize[j][1]):
					augmantdata[count1].append(sdata[i][x][y])
			count1+=1
	augmantdata=np.array(augmantdata,dtype=float)
	augmantdata/=255.0
	augmantdata-=mean
	augmantdata/=std
	augmantdata=np.reshape(augmantdata,(5*len(sdata),42,42,1))
	ans1=model.predict(augmantdata,batch_size=5*len(sdata))
	print(ans1)
	print(ans1.shape)
	del sdata
	del augmantdata
	for i in range(0,len(ans1),5):
		max=0
		maxnum=0
		sum=[0,0,0,0,0,0,0]

		for j in range(len(ans1[0])):
			for k in range(5):
				sum[j]+=ans1[i+k][j]
			sum[j]/=5
			if(sum[j]>max):
				max=sum[j]
				maxnum=j
		output.append(maxnum)
sdata=[]
for i in range((cut-1)*leng,len(data)):
	sdata.append(data[i][0].split())
sdata=np.array(sdata,dtype=float)
sdata=np.reshape(sdata,(len(sdata),48,48))
augmantdata=[]
crosssize=[[0,0],[0,6],[6,0],[6,6],[3,3]]
count1=0
for i in range(len(sdata)):
	for j in range(5):
		augmantdata.append([])
		for x in range(crosssize[j][0],42+crosssize[j][0]):
			for y in range(crosssize[j][1],42+crosssize[j][1]):
				augmantdata[count1].append(sdata[i][x][y])
		count1+=1
augmantdata=np.array(augmantdata,dtype=float)
augmantdata/=255.0
augmantdata-=mean
augmantdata/=std
augmantdata=np.reshape(augmantdata,(5*len(sdata),42,42,1))
ans1=model.predict(augmantdata,batch_size=5*len(sdata))
print(ans1)
print(ans1.shape)
del sdata
del augmantdata
for i in range(0,len(ans1),5):
	max=0
	maxnum=0
	sum=[0,0,0,0,0,0,0]
	for j in range(len(ans1[0])):
		for k in range(5):
			sum[j]+=ans1[i+k][j]
		sum[j]/=5
		if(sum[j]>max):
			max=sum[j]
			maxnum=j
	output.append(maxnum)
correct=0
with open(sys.argv[2], 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(["id","label"])
	for i in range(len(output)):
		writer.writerow([str(i),output[i]])
#		python HW8_test.py test.csv ans.csv