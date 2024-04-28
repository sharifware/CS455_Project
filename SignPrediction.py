

'''
Road Lane Detection
Created by: Muhammed-Sharif Adepetu
Used for: CS455 Final Project

Creates a CNN model to determine if an image is either a traffic sign or stoplight.
it then trains a model
Finally the model is tested using the test data to determine the outputs

Data implementation must be in a folder titled sign_data
Then in that split into a test, train, and valication folder
Inside of each of those have a stop_sign and fraffic light folder

These consist of the data utilized to test the model properlly

'''import os, glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.layers import Dropout,Flatten,Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import to_categorical

w=56
work_dir='sign_data/traffic_signs'
npy_data_base='traffic_signs-data'
npy_labels_base='traffic_signs-labels'
classes=['Light','Stop']
num_classes=len(classes)
model_file="traffic_signs-model-w{0}.h5".format(w)
history_file="traffic_signs-history-w{0}.csv".format(w)
pred_dir=work_dir+"/predicted"
results_base='traffic_signs-results'
model_file="traffic_signs-model-w{0}.h5".format(w)
classes=["Light","Stop"]
num_classes=len(classes)

ok=1
for mode in ["train", "test"]:
    X=[]
    y=[]
    npy_data_file='{0}-{1}-w{2}.npy'.format(npy_data_base,mode,w)
    npy_labels_file='{0}-{1}-w{2}.npy'.format(npy_labels_base,mode,w)
    for i in range(0, num_classes):
        cls=classes[i]
        image_dir=work_dir+"/original/"+mode+"/"+cls
        files=glob.glob(image_dir+"/*.*")
        print("w:",w,"mode:",mode,"class:",cls)
        for f in files:
            img=Image.open(f)
            img=img.convert("RGB")
            img=img.resize((w,w))
            data=np.array(img)
            X.append(data)
            y.append(i)
    X=np.array(X)
    y=np.array(y)
    data_file=work_dir+"/"+npy_data_file
    labels_file=work_dir+"/"+npy_labels_file
    np.save(data_file,X)
    np.save(labels_file,y)
    
    if not os.path.exists(data_file):
        ok=0
    if not os.path.exists(labels_file):
        ok=0
        
if ok==1:
    print("OK")

mode="train"
npy_base_file="{0}/{1}-{2}-w{3}.npy".format(work_dir,npy_data_base,mode,w)
npy_labels_file="{0}/{1}-{2}-w{3}.npy".format(work_dir,npy_labels_base,mode,w)
X_train=np.load(npy_base_file).astype("float16")
X_train /= 255
y_train=np.load(npy_labels_file)
y_train=to_categorical(y_train,num_classes)

mode="test"
npy_base_file="{0}/{1}-{2}-w{3}.npy".format(work_dir,npy_data_base,mode,w)
npy_labels_file="{0}/{1}-{2}-w{3}.npy".format(work_dir,npy_labels_base,mode,w)
X_test=np.load(npy_base_file).astype("float16")
X_test/=255
y_test=np.load(npy_labels_file)
y_test=to_categorical(y_test,num_classes)

print( X_train.shape )
print( y_train.shape )
print( X_test.shape )
print( y_test.shape )
img_rows=X_train.shape[1]
img_cols=X_train.shape[2]
img_channels=X_train.shape[3]
print("image_size:", img_rows, img_cols)
print("image_channels:", img_channels)

input_shape=(img_rows, img_cols, img_channels)
model=Sequential()
model.add(Conv2D(16, (3,3), activation='relu', padding='same',
input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
...
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

input_shape=(img_rows, img_cols, img_channels)
model=Sequential()
model.add(Conv2D(16, (3,3), activation='relu', padding='same',
input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

n_epochs=40
val_split=0.2
batch_size=128
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
metrics=['accuracy'])
cl=CSVLogger(history_file)
es=EarlyStopping(monitor='val_loss',patience=19,verbose=1)
fit_log=model.fit(X_train, y_train, batch_size=batch_size,
epochs=n_epochs, validation_split=val_split,
callbacks=[cl, es])

model.save(model_file)

#Open a file for saving results
results_file='{0}-w{1}.csv'.format(results_base,w)
res=open(results_file,'w')
res.write("file,{0},{1}\n".format(classes[0],classes[1]))

files=glob.glob(pred_dir+"/*.*")
for f in files:
    os.remove(f)

#Do prediction for each image file
image_dir=work_dir+"/original/unknown"
files=glob.glob(image_dir+"/*.*")
for f in files:
    #Load image data
    img=Image.open(f)
    img=img.convert("RGB")
    img=img.resize((w,w))
    #reshape 4D numpy array
    X=np.asarray(img).reshape(1,w,w,3).astype("float16")
    X_pred=X/255
    a = np.array( [ [ 0, 1, 2 ], [ 3, 4, 5 ] ] )
    print( a.shape )
    print( a )
    b = a.reshape( 3, 2 )
    print( b.shape )
    print( b )
    c = np.array( [ 0, 1, 2, 3, 4, 5 ] )
    print( c.shape )
    c2d = c.reshape( 1, 6 )
    print( c2d.shape )
    print( c2d )
    #Classification prediction
    pred=model.predict(X_pred)
    #Output image to pred_dir
    pred_ans=pred.argmax()
    pred_cls=classes[pred_ans]
    tag="pred_as_{0}-w{1}".format(pred_cls,w)
    base=os.path.basename(f)
    img_file="{0}/{1}.{2}.jpg".format(pred_dir,os.path.splitext(base)[0],tag)
    img.save(img_file)
    print(img_file,pred)
    res.write("{0},{1},{2}\n".format(f,pred[0][0],pred[0][1]))
    #Close the result file
    
res.close()
