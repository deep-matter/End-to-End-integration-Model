data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        #layers.RandomZoom(.5, .2)
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        layers.Rescaling(1./255)
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)



datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=0.2, # rotation
        width_shift_range=0.2, # horizontal shift
        height_shift_range=0.2, # vertical shift
        zoom_range=0.2, # zoom
        horizontal_flip=True, # horizontal flip
        brightness_range=[0.2,1.2]) 


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)


train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')            

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
losse=tf.keras.losses.CategoricalCrossentropy(
     label_smoothing=0.1,
    name='categorical_crossentropy'
)
   
model.compile(optimizer='adam', loss=losse, metrics=['accuracy'])   


data_augmentation = keras.Sequential(
    [
       
        layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)


ef New_model(n_class):
    model_name = 'CNN_12'

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(128,128,1)))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Convolution2D(32, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Convolution2D(64, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(128, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Convolution2D(128, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(256, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(BatchNormalization())
    model.add(PReLU())    

    model.add(Convolution2D(256, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(512, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Convolution2D(512, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(BatchNormalization())
    model.add(PReLU())    
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(1028, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(BatchNormalization())
    model.add(PReLU())
    
    model.add(Convolution2D(1028, 3, 3))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(AveragePooling2D(pool_size=(2,2)))

    model.add(Flatten())
    # Dense = Fully connected layer
    model.add(Dense(n_class))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy']
                 )
#orthe model fix =================================================


inp = keras.Input(shape=(224, 224, 3))
m = (Conv2D(64, (3, 3), padding='same', activation='relu'))(inp)
m = (BatchNormalization())(m)
m = (Activation('linear'))(m)
mod = (Conv2D(128, (1, 1), strides=2))(m)
mod = (BatchNormalization())(mod)

m = (SeparableConv2D(128, (3, 3), padding='same', activation='relu'))(m)
m = (BatchNormalization())(m)
m = (Activation('linear'))(m)
m = (MaxPool2D((3, 3), 2, padding='same'))(m)
m = keras.layers.Add()([m, mod])
mod = (Conv2D(256, (1, 1), strides=2))(m)
mod = (BatchNormalization())(mod)

m = (SeparableConv2D(256, (3, 3), padding='same', activation='relu'))(m)
m = (BatchNormalization())(m)
m = (Activation('linear'))(m)
m = (MaxPool2D((3, 3), 2, padding='same'))(m)
m = keras.layers.Add()([m, mod])
mod = (Conv2D(784, (1, 1), strides=2))(m)
mod = (BatchNormalization())(mod)

m = (SeparableConv2D(784, (3, 3), padding='same', activation='relu'))(m)
m = (BatchNormalization())(m)
m = (Activation('linear'))(m)
m = (MaxPool2D((3, 3), 2, padding='same'))(m)
m = keras.layers.Add()([m, mod])
mod = (Conv2D(784, (1, 1), strides=2))(m)
mod = (BatchNormalization())(mod)

m = (SeparableConv2D(784, (3, 3), padding='same', activation='relu'))(m)
m = (BatchNormalization())(m)
m = (Activation('linear'))(m)
m = (MaxPool2D((3, 3), 2, padding='same'))(m)
m = keras.layers.Add()([m, mod])
mod = (Conv2D(1024, (1, 1), strides=2))(m)
mod = (BatchNormalization())(mod)

m = (SeparableConv2D(1024, (3, 3), padding='same', activation='relu'))(m)
m = (BatchNormalization())(m)
m = (Activation('linear'))(m)
m = (MaxPool2D((3, 3), 2, padding='same'))(m)
m = keras.layers.Add()([m, mod])
mod = (Conv2D(1024, (1, 1), strides=2))(m)
mod = (BatchNormalization())(mod)

m = (SeparableConv2D(1024, (3, 3), padding='same', activation='relu'))(m)
m = (BatchNormalization())(m)
m = (Activation('linear'))(m)
m = (MaxPool2D((3, 3), 2, padding='same'))(m)
m = keras.layers.Add()([m, mod])
mod = (Conv2D(784, (1, 1), strides=2))(m)
mod = (BatchNormalization())(mod)

m = (SeparableConv2D(784, (3, 3), padding='same', activation='relu'))(m)
m = (BatchNormalization())(m)
m = (Activation('linear'))(m)
m = (MaxPool2D((3, 3), 2, padding='same'))(m)
m = keras.layers.Add()([m, mod])
mod = (Conv2D(512, (1, 1), strides=2))(m)
mod = (BatchNormalization())(mod)

m = (SeparableConv2D(512, (3, 3), padding='same', activation='relu'))(m)
m = (BatchNormalization())(m)
m = (Activation('linear'))(m)
m = (MaxPool2D((3, 3), 2, padding='same'))(m)
m = keras.layers.Add()([m, mod])
mod = (Conv2D(256, (1, 1), strides=2))(m)
mod = (BatchNormalization())(mod)

m = (SeparableConv2D(256, (3, 3), padding='same', activation='relu'))(m)
m = (BatchNormalization())(m)
m = (Activation('linear'))(m)
m = (MaxPool2D((3, 3), 2, padding='same'))(m)
m = keras.layers.Add()([m, mod])

m = (SeparableConv2D(128, (3, 3), padding='same', activation='relu'))(m)
m = (BatchNormalization())(m)
m = (Activation('linear'))(m)
m = (SeparableConv2D(128, (3, 3), padding='same', activation='relu'))(m)
m = (BatchNormalization())(m)
m = (Activation('linear'))(m)
m = (Dropout(0.2))(m)
m = (keras.layers.GlobalMaxPooling2D())(m)
m = (Dense(1, activation='sigmoid'))(m)
model = keras.Model(inp, m)

history = model.fit_generator(imgen.flow(x_t, y_t, batch_size=16), epochs=35, steps_per_epoch=len(x_t)//16, validation_data=(x_v, y_v), validation_steps=len(x_v)//16, shuffle=True)

from keras.optimizers import Adadelta
from keras.utils import to_categorical

model.compile(loss= 'binary_crossentropy', optimizer=Adadelta(lr=0.01), metrics=['accuracy'])

#=====================================================================

# Initiate the train and test generators with data Augumentation
sometimes = lambda imgaug: iaa.Sometimes(0.6, aug)
seq = iaa.Sequential([
                      iaa.GaussianBlur(sigma=(0 , 1.0)),
                      iaa.Sharpen(alpha=1, lightness=0),
                      iaa.CoarseDropout(p=0.1, size_percent=0.15),
                              sometimes(iaa.Affine(
                                                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                                    rotate=(-30, 30),
                                                    shear=(-16, 16)))
                    ])

y = Conv2D(filters=filters_1,kernel_size= kernel_size,activation='relu')(augmented)
y = ZeroPadding2D(padding=(1,1))(y)

y = Conv2D(filters=filters_1,kernel_size=kernel_size, activation='relu')(y)
y = MaxPooling2D(pool_size=(2,2))(y)

y = Conv2D(filters=filters_2,kernel_size=kernel_size,activation='relu')(y)
y = ZeroPadding2D(padding=(1,1))(y)
y = Dropout(dropout)(y) 


y = Conv2D(filters=filters_2,kernel_size=kernel_size, activation='relu')(y)
y = MaxPooling2D(pool_size=(2,2))(y)
y = BatchNormalization()(y)

y = Conv2D(filters=filters_3,kernel_size=kernel_size,activation='relu')(y)
y = ZeroPadding2D(padding=(1,1))(y)



y = Conv2D(filters=filters_3,kernel_size=kernel_size,activation='relu')(y)
y = MaxPooling2D(pool_size=(2,2))(y)
y = Dropout(dropout)(y)

y = Conv2D(filters=filters_4,kernel_size=kernel_size,activation='relu')(y)
y = MaxPooling2D(pool_size=(2,2))(y)
y = BatchNormalization()(y)

y = Conv2D(filters=filters_4, kernel_size=kernel_size,activation='relu')(y)
# image to vector before connecting to dense layer
y = Flatten()(y)
# dropout regularization
y = Dropout(dropout)(y)
outputs = Dense(len(num_labels), activation='softmax')(y)
#======================================================

# use functional API to build cnn layers
inputs = Input(shape=input_shape)
augmented = data_augmentation(inputs)


y = Conv2D(first_filters,kernel_size,activation='relu')(augmented)
y = Conv2D(first_filters,kernel_size,activation='relu')(y)
y = Conv2D(first_filters,kernel_size,activation='relu')(y)
y = MaxPool2D(pool_size=pool_size)(y)
y = Dropout(dropout_conv)(y) 


y = Conv2D(second_filters,kernel_size,activation='relu')(y)
y = Conv2D(second_filters,kernel_size,activation='relu')(y)
y = Conv2D(second_filters,kernel_size,activation='relu')(y)
y = MaxPool2D(pool_size=pool_size)(y)
y = Dropout(dropout_conv)(y) 


y = Conv2D(third_filters,kernel_size,activation='relu')(y)
y = Conv2D(third_filters,kernel_size,activation='relu')(y)
y = Conv2D(third_filters,kernel_size,activation='relu')(y)
y = MaxPool2D(pool_size=pool_size)(y)
y = Dropout(dropout_conv)(y)


y = GlobalAveragePooling2D()(y)
# dropout regularization
y = Dense(256, activation='relu')(y)
y = Dropout(dropout_dense)(y)
outputs = Dense(len(num_labels), activation='softmax')(y)
#=====================================================

data = np.array(data, dtype="float")/255.0
labels = np.array(labels,dtype ="uint8")

(trainX, testX, trainY, testY) = train_test_split(
                                data,labels, 
                                test_size=0.2, 
                                random_state=42) 

train_datagen = keras.preprocessing.image.ImageDataGenerator(
          zoom_range = 0.1,
          width_shift_range = 0.2, 
          height_shift_range = 0.2,
          horizontal_flip = True,
          fill_mode ='nearest') 

 image_generator = ImageDataGenerator(
            rotation_range=10,
            zoom_range=1.1,
            width_shift_range=0.07,
            height_shift_range=0.07,
            brightness_range=[0.2,1.0],
            shear_range=0.25,
            horizontal_flip=False,
            vertical_flip=False,
            data_format="channels_last")

val_datagen = keras.preprocessing.image.ImageDataGenerator()


train_generator = train_datagen.flow(
        trainX, 
        trainY,
        batch_size=batch_size,
        shuffle=True)

validation_generator = val_datagen.flow(
                testX,
                testY,
                batch_size=batch_size) 