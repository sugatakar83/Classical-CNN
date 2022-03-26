import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import layers
from keras import optimizers
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

# load data
rawdata = np.loadtxt('fashion-mnist_train.csv', dtype=int, delimiter=',', skiprows=1)
# split labels and pixel values
X = rawdata[:, 1:]
y = rawdata[:, 0]
# convert pixel values to 2d arrays.Images are converted into Numpy Array in Height, Width, Channel format
X = np.reshape(X, (-1, 28, 28))
# one hot encode labels. Converts the class vector (integers) to binary class matrix
num_classes = 10
y_oh = to_categorical(y, num_classes)

# scale pixel values to be between 0 and 1
X_scaled = X / 255
X_scaled = np.expand_dims(X_scaled, -1)  # channels last

# split data into train set and balanced validation set
num_val = int(y.shape[0] * 0.1)
validation_mask = np.zeros(y.shape[0], np.bool)
np.random.seed(1)
for c in range(num_classes):
    idxs = np.random.choice(np.flatnonzero(y == c), num_val // 10, replace=False)
    validation_mask[idxs] = 1
np.random.seed(None)

X_train = X_scaled[~validation_mask]
X_val = X_scaled[validation_mask]
print("Train/val pixel shapes:", X_train.shape, X_val.shape)

y_train = y_oh[~validation_mask]
y_val = y_oh[validation_mask]


# Inception Modules
def conv2D_bn_relu(x, filters, kernel_size, strides, padding='valid', kernel_initializer='glorot_uniform', name=None):
    """2D convolution with batch normalization and ReLU activation.
    """

    x = layers.Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      kernel_initializer=kernel_initializer,
                      name=name,
                      use_bias=False)(x)
    x = layers.BatchNormalization(scale=False)(x)
    return layers.Activation('relu')(x)


def inception_module_A(x, filters=None, kernel_initializer='glorot_uniform'):
    """Inception module A as described in Figure 4 of "Inception-v4, Inception-ResNet
    and the Impact of Residual Connections on Learning" (Szegedy, et al. 2016).

    # Arguments
        x: 4D tensor with shape: `(batch, rows, cols, channels)`.
        filters: Number of output filters for the module.
        kernel_initializer: Weight initializer for all convolutional layers in module.
    """

    if filters is None:
        filters = int(x.shape[-1])
    branch_filters = filters // 4

    b1 = conv2D_bn_relu(x,
                        filters=(branch_filters // 3) * 2,
                        kernel_size=1,
                        strides=1,
                        kernel_initializer=kernel_initializer)
    b1 = conv2D_bn_relu(b1,
                        filters=branch_filters,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer=kernel_initializer)

    b2 = conv2D_bn_relu(x,
                        filters=(branch_filters // 3) * 2,
                        kernel_size=1,
                        strides=1,
                        kernel_initializer=kernel_initializer)
    b2 = conv2D_bn_relu(b2,
                        filters=branch_filters,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer=kernel_initializer)
    b2 = conv2D_bn_relu(b2,
                        filters=branch_filters,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer=kernel_initializer)

    b3 = conv2D_bn_relu(x,
                        filters=branch_filters,
                        kernel_size=1,
                        strides=1,
                        kernel_initializer=kernel_initializer)

    pool = layers.AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
    pool = conv2D_bn_relu(pool,
                          filters=branch_filters,
                          kernel_size=1,
                          strides=1,
                          kernel_initializer=kernel_initializer)

    return layers.concatenate([b1, b2, b3, pool])


def inception_module_C(x, filters=None, kernel_initializer='glorot_uniform'):
    """Inception module C as described in Figure 6 of "Inception-v4, Inception-ResNet
    and the Impact of Residual Connections on Learning" (Szegedy, et al. 2016).

    # Arguments
        x: 4D tensor with shape: `(batch, rows, cols, channels)`.
        filters: Number of output filters for the module.
        kernel_initializer: Weight initializer for all convolutional layers in module.
    """

    if filters is None:
        filters = int(x.shape[-1])
    branch_filters = filters // 6

    b1 = conv2D_bn_relu(x,
                        filters=(branch_filters // 2) * 3,
                        kernel_size=1,
                        strides=1,
                        kernel_initializer=kernel_initializer)

    b1a = conv2D_bn_relu(b1,
                         filters=branch_filters,
                         kernel_size=(1, 3),
                         strides=1,
                         padding='same',
                         kernel_initializer=kernel_initializer)

    b1b = conv2D_bn_relu(b1,
                         filters=branch_filters,
                         kernel_size=(3, 1),
                         strides=1,
                         padding='same',
                         kernel_initializer=kernel_initializer)

    b2 = conv2D_bn_relu(x,
                        filters=(branch_filters // 2) * 3,
                        kernel_size=1,
                        strides=1,
                        kernel_initializer=kernel_initializer)
    b2 = conv2D_bn_relu(b2,
                        filters=(branch_filters // 4) * 7,
                        kernel_size=(1, 3),
                        strides=1,
                        padding='same',
                        kernel_initializer=kernel_initializer)
    b2 = conv2D_bn_relu(b2,
                        filters=branch_filters * 2,
                        kernel_size=(3, 1),
                        strides=1,
                        padding='same',
                        kernel_initializer=kernel_initializer)

    b2a = conv2D_bn_relu(b2,
                         filters=branch_filters,
                         kernel_size=(1, 3),
                         strides=1,
                         padding='same',
                         kernel_initializer=kernel_initializer)

    b2b = conv2D_bn_relu(b2,
                         branch_filters,
                         kernel_size=(3, 1),
                         strides=1,
                         padding='same',
                         kernel_initializer=kernel_initializer)

    b3 = conv2D_bn_relu(x,
                        filters=branch_filters,
                        kernel_size=1,
                        strides=1,
                        kernel_initializer=kernel_initializer)

    pool = layers.AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
    pool = conv2D_bn_relu(pool,
                          filters=branch_filters,
                          kernel_size=1,
                          strides=1,
                          kernel_initializer=kernel_initializer)

    return layers.concatenate([b1a, b1b, b2a, b2b, b3, pool])


def reduction_module_A(x, filters, kernel_initializer='glorot_uniform'):
    """Reduction module A as described in Figure 7 of "Inception-v4, Inception-ResNet
    and the Impact of Residual Connections on Learning" (Szegedy, et al. 2016).

    # Arguments
        x: 4D tensor with shape: `(batch, rows, cols, channels)`.
        filters: Number of output filters for the module.
        kernel_initializer: Weight initializer for all convolutional layers in module.
    """

    branch_filters = (filters - int(x.shape[-1])) // 2

    b1 = conv2D_bn_relu(x,
                        filters=branch_filters,
                        kernel_size=3,
                        strides=2,
                        padding='same',
                        kernel_initializer=kernel_initializer)

    b2 = conv2D_bn_relu(x,
                        filters=(branch_filters // 3) * 2,
                        kernel_size=1,
                        strides=1,
                        kernel_initializer=kernel_initializer)
    b2 = conv2D_bn_relu(b2,
                        filters=(branch_filters // 6) * 5,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer=kernel_initializer)
    b2 = conv2D_bn_relu(b2,
                        filters=branch_filters,
                        kernel_size=3,
                        strides=2,
                        padding='same',
                        kernel_initializer=kernel_initializer)

    pool = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    return layers.concatenate([b1, b2, pool])

# Inception Model
K.clear_session()

stem_width = 128

inputs = layers.Input(shape=X_scaled.shape[1:])
x = conv2D_bn_relu(inputs,
                   filters=stem_width,
                   kernel_size=5,
                   strides=1,
                   padding='same',
                   name='conv_1')

x = reduction_module_A(x, filters=int(2*stem_width))
# Spatial Dropouts performs the same function as Dropout, however, it drops entire 2D feature maps
# instead of individual elements. If adjacent pixels within feature maps are strongly correlated
# (as is normally the case in early convolution layers) then regular dropout will not regularize the activations
# and will otherwise just result in an effective learning rate decrease. In this case,
# SpatialDropout2D will help promote independence between feature maps and should be used instead.
x = layers.SpatialDropout2D(0.3)(x)

x = inception_module_A(x, filters=int(2*stem_width))
x = inception_module_A(x, filters=int(2*stem_width))

x = reduction_module_A(x, filters=int(3*stem_width))
x = layers.SpatialDropout2D(0.5)(x)

x = inception_module_C(x, filters=int(3*stem_width))
x = inception_module_C(x, filters=int(3*stem_width))
# Global Avg Pooling is used to generate one feature map for each corresponding category of the
# classification task in the last convolution layer
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(num_classes, name='logits')(x)
x = layers.Activation('softmax', name='softmax')(x)

model = Model(inputs=inputs, outputs=x)

# Label Smoothing.By applying label smoothing we can lessen the confidence of the model and
# prevent it from descending into deep crevices of the loss landscape where overfitting occurs.
epsilon = 0.001
y_train_smooth = y_train * (1 - epsilon) + epsilon / 10

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adamax(lr=0.006, beta_1=0.49, beta_2=0.999),
              metrics=['accuracy'])
# define data augmentations
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# train model
batch_size = 128
epochs = 10

history = model.fit_generator(
    datagen.flow(X_train, y_train_smooth, batch_size=batch_size, shuffle=True),
    epochs=epochs,
    steps_per_epoch=(len(y_train) - 1) // batch_size + 1,
    validation_data=(X_val, y_val)
)
