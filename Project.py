# Import necessary modules from Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the model architecture
model = Sequential()
# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01), input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# Flatten the tensor output from the convolutional layers
model.add(Flatten())
# Add dense layers
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define data generators for training and validation data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

# Define data generators
train_generator = train_datagen.flow_from_directory(
        'data/train', #directory for training data
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        'data/validation', #directory for validation data
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Define test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

# Define test generator
test_generator = test_datagen.flow_from_directory(
        'data/test',  #directory for test data
        target_size=(64, 64),
        batch_size=1,
        class_mode='binary',
        shuffle=False)

# Calculate steps per epoch
steps_per_epoch = train_generator.samples // train_generator.batch_size

# Train the model
history = model.fit(
      train_generator,
      batch_size=32,
      steps_per_epoch=steps_per_epoch,
      epochs=100,
      validation_data=validation_generator,
      validation_split=0.2,
      callbacks=[early_stopping])

# Print the accuracy
print("Training Accuracy: ", history.history['accuracy'][-1])
print("Validation Accuracy: ", history.history['val_accuracy'][-1])

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Get the filenames from the generator
fileNames = test_generator.filenames

# Get the true labels
true_labels = test_generator.classes

# Get the label to class mapping from the generator
label2index = test_generator.class_indices

# Getting the mapping from class index to class label
class_label = dict((v,k) for k,v in label2index.items())

# Have the model complete its predictions
predictions = model.predict(test_generator, steps=int(test_generator.samples/test_generator.batch_size),verbose=1)
predicted_classes = (predictions > 0.5).astype(int).flatten()

corrects = np.where(predicted_classes == true_labels)[0]
print("Number of correct predictions = {}/{}".format(len(corrects),test_generator.samples))

# Plot out the images with correct predictions
for i in range(len(corrects)):
    pred_class = predicted_classes[corrects[i]]
    pred_label = class_label[pred_class]
    
    confidence = predictions[corrects[i]][0]

    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fileNames[corrects[i]].split('/')[0],
        pred_label,
        confidence)
    
    original = image.load_img('{}/{}'.format('data/test',fileNames[corrects[i]]))
    plt.figure(figsize=[7,7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()

errors = np.where(predicted_classes != true_labels)[0]
print("Number of errors = {}/{}".format(len(errors),test_generator.samples))

# Plot out the images with errors
for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = class_label[pred_class]
    
    confidence = predictions[errors[i]][0]

    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fileNames[errors[i]].split('/')[0],
        pred_label,
        confidence)
    
    original = image.load_img('{}/{}'.format('data/test',fileNames[errors[i]]))
    plt.figure(figsize=[7,7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()