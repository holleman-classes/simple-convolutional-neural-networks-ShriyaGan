from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout, MaxPool2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    # plt.show()

def predict_image_class(model, image_path, class_names):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(32, 32))
    image = img_to_array(image)  # Convert the image to a numpy array
    image = image.astype('float32') / 255  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add a batch dimension

    # Make a prediction
    predictions = model.predict(image)

    # Interpret the prediction
    predicted_class = np.argmax(predictions, axis=1)
    predicted_class_name = class_names[predicted_class[0]]
    print(f'Predicted class: {predicted_class_name}')

def build_model1():
    model = Sequential([
        Conv2D(32, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', input_shape=(32,32,3)),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(4,4), strides=(4,4)),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
    return model

def build_model2():
    model = Sequential([
        Conv2D(32, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', input_shape=(32,32,3)),
        BatchNormalization(),
        SeparableConv2D(64, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
        BatchNormalization(),
        SeparableConv2D(128, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
        BatchNormalization(),
        SeparableConv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
        BatchNormalization(),
        SeparableConv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
        BatchNormalization(),
        SeparableConv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
        BatchNormalization(),
        SeparableConv2D(128, kernel_size=(3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(4,4), strides=(4,4)),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
    return model

def build_model3():
    input_layer = Input(shape=(32,32,3))

    conv1 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2))(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.25)(conv1)
    
    conv2 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.25)(conv2)

    conv3 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2))(conv2)
    conv3 = BatchNormalization()(conv3)
    
    skip1 = Conv2D(128, kernel_size=(1,1), strides=(4,4))(conv1)
    skip1 = layers.add([conv3, skip1])
    skip1 = Dropout(0.25)(skip1)

    conv4 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2))(skip1)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.25)(conv4)

    conv5 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2))(conv4)
    conv5 = BatchNormalization()(conv5)

    skip2 = layers.add([conv5, skip1])
    skip2 = Dropout(0.25)(skip2)

    conv6 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2))(skip2)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(0.25)(conv6)

    conv7 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2))(conv6)
    conv7 = BatchNormalization()(conv7)

    skip3 = layers.add([conv7, skip2])
    skip3 = Dropout(0.25)(skip3)

    pooling = MaxPool2D(pool_size=(4,4), strides=(4,4))(skip3)
    flatten = Flatten()(pooling)
    dense = Dense(128, activation='relu')(flatten)
    dense = BatchNormalization()(dense)
    dense = Dense(10, activation='relu')(dense)

    model = models.Model(inputs=input_layer, outputs=dense)
    return model

def build_model50k():
    model = Sequential([
        Conv2D(16, (3, 3), padding='same', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        GlobalAveragePooling2D(),

        Dense(120, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# no training or dataset construction should happen above this line
if __name__ == '__main__':

##############  Setup Cifar10 Dataset  ############## 
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    train_labels = train_labels.squeeze()
    test_labels = test_labels.squeeze()

    train_images = train_images / 255.0
    test_images  = test_images  / 255.0

  # Split training data into training and validation sets
    val_split = 0.2
    val_size = int(len(train_images) * val_split)
    train_images, test_images = train_images[:-val_size], train_images[-val_size:]
    train_labels, test_labels = train_labels[:-val_size], train_labels[-val_size:]


################ MODEL 1 #################
# Build, compile and train model 1
model1 = build_model1()
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model1.summary()

# # VALIDATION #
# # history1 = model1.fit(train_images, train_labels, epochs=50, validation_split=0.2)
# # test_loss1, test_acc1 = model1.evaluate(test_images, test_labels)
# # training_acc1 = history1.history['accuracy'][-1]
# # validation_acc1 = history1.history['val_accuracy'][-1]
# # print(f'Model 1 Training accuracy: {training_acc1}, Test accuracy: {test_acc1}, Validation accuracy: {validation_acc1}')

# # # Plot the training history
# # plot_training_history(history1)
# # plt.savefig('Training and Validation for model 1.png')

# # image_path = 'test_image_dog.png' 
# # predict_image_class(model1, image_path, class_names)

  
################ MODEL 2 ###################
# Build, compile, and train model 2 (Depthwise Separable Convolutions)
model2 = build_model2()
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.summary()

# # # VALIDATION #
# # history2 = model2.fit(train_images, train_labels, epochs=50, validation_split=0.2)
# # test_loss2, test_acc2 = model2.evaluate(test_images, test_labels)
# # training_acc2 = history2.history['accuracy'][-1]
# # validation_acc2 = history2.history['val_accuracy'][-1]
# # print(f'Model 2 Training accuracy: {training_acc2}, Test accuracy: {test_acc2}, Validation accuracy: {validation_acc2}')

# # # Plot the training history
# # plot_training_history(history2)
# # plt.savefig('Training and Validation for model 2.png')

# # image_path = 'test_image_dog.png'  
# # predict_image_class(model2, image_path, class_names)


################ MODEL 3 ###################
# Build, compile, and train model 2 (Depthwise Separable Convolutions)
model3 = build_model3()
model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model3.summary()

# # VALIDATION #
# history3 = model3.fit(train_images, train_labels, epochs=50, validation_split=0.2)
# test_loss3, test_acc3 = model3.evaluate(test_images, test_labels)
# training_acc3 = history3.history['accuracy'][-1]
# validation_acc3 = history3.history['val_accuracy'][-1]
# print(f'Model 3 Training accuracy: {training_acc3}, Test accuracy: {test_acc3}, Validation accuracy: {validation_acc3}')

# # Plot the training history
# plot_training_history(history3)
# plt.savefig('Training and Validation for model 3.png')

# image_path = 'test_image_dog.png' 
# predict_image_class(model3, image_path, class_names)

############### MODEL 50K ###################
# Build, compile, and train model 2 (Depthwise Separable Convolutions)
model50k = build_model50k()
model50k.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model50k.summary()
# history50k = model50k.fit(train_images, train_labels, epochs=50, validation_split=0.2)
# model50k.save("best_model.h5")

# # VALIDATION #
# test_loss50k, test_acc50k = model50k.evaluate(test_images, test_labels)
# training_acc50k = history50k.history['accuracy'][-1]
# validation_acc50k = history50k.history['val_accuracy'][-1]
# print(f'Model 3 Training accuracy: {training_acc50k}, Test accuracy: {test_acc50k}, Validation accuracy: {validation_acc50k}')

# # Plot the training history
# plot_training_history(history50k)
# plt.savefig('Training and Validation for model 50k.png')

# image_path = 'test_image_dog.png'  
# predict_image_class(model50k, image_path, class_names)


  
