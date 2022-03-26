import tensorflow as tf
import time
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Add
from tensorflow.keras import Model

# get the F-MNIST data from tensorflow dataset
dataset, info = tfds.load('fashion_mnist', as_supervised = True, with_info = True)
dataset_test, dataset_train = dataset['test'], dataset['train']

# Training data is reshaped into a numpy array and divided by 255 so values fit between 0 and 1
def convert_types(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

batch_size = 128

# create the test and train datasets
dataset_train = dataset_train.map(convert_types).shuffle(10000).batch(batch_size)
dataset_test = dataset_test.map(convert_types).batch(batch_size)

# data augmentation
datagen = ImageDataGenerator(rotation_range = 10, horizontal_flip = True, zoom_range = 0.1)


# Algorithm for the residual block:
# X_skip = Input
# Convolutional Layer (1X1) (Padding=’same’) (Filters = f) →(Input)
# Batch Normalisation →(Input)
# Relu Activation →(Input)
# Convolutional Layer (3X3) (Padding = ‘same’) (Filters = f) →(Input)
# Batch Normalisation →(Input)
# Add (Input + X_skip)
# Relu Activation

class ResidualBlock(Model):
    def __init__(self, channel_in=64, channel_out=256):
        super().__init__()

        channel = channel_out // 4

        self.conv1 = Conv2D(channel, kernel_size=(1, 1), padding="same")
        self.bn1 = BatchNormalization()
        self.av1 = Activation(tf.nn.relu)
        self.conv2 = Conv2D(channel, kernel_size=(3, 3), padding="same")
        self.bn2 = BatchNormalization()
        self.av2 = Activation(tf.nn.relu)
        self.conv3 = Conv2D(channel_out, kernel_size=(1, 1), padding="same")
        self.bn3 = BatchNormalization()
        self.shortcut = self._shortcut(channel_in, channel_out)
        self.add = Add()
        self.av3 = Activation(tf.nn.relu)

    def call(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.av1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.av2(h)
        h = self.conv3(h)
        h = self.bn3(h)
        shortcut = self.shortcut(x)
        h = self.add([h, shortcut])
        y = self.av3(h)
        return y

    def _shortcut(self, channel_in, channel_out):
        if channel_in == channel_out:
            return lambda x: x
        else:
            return self._projection(channel_out)

    def _projection(self, channel_out):
        return Conv2D(channel_out, kernel_size=(1, 1), padding="same")

# ResNet50 have 6 layers
# 1. convolution layer 7X7 kernel, stride 2, nos of filters is 32
# 2. 2nd layer starts with 3X3 maxpool with stride 2 followed by convolution layer of:
#    [(1X1,64), (3X3, 64), (1X1, 256)] X 3
# 3. convolution layer of:
#    [(1X1,128), (3X3, 128), (1X1, 512)] X 4
# 4. convolution layer of:
#    [(1X1,256), (3X3, 256), (1X1, 1024)] X 6
# 5. convolution layer of:
#    [(1X1,512), (3X3, 512), (1X1, 2048)] X 3
# 6. consider average pooling

class ResNet50(Model):
    def __init__(self, input_shape, output_dim):
        super().__init__()

        self._layers = [
            # conv1
            Conv2D(64, input_shape=input_shape, kernel_size=(7, 7), strides=(2, 2), padding="same"),
            BatchNormalization(),
            Activation(tf.nn.relu),
            # conv2_x
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
            ResidualBlock(64, 256),
            [
                ResidualBlock(256, 256) for _ in range(2)
            ],
            # conv3_x
            Conv2D(512, kernel_size=(1, 1), strides=(2, 2)),
            [
                ResidualBlock(512, 512) for _ in range(4)
            ],
            # conv4_x
            Conv2D(1024, kernel_size=(1, 1), strides=(2, 2)),
            [
                ResidualBlock(1024, 1024) for _ in range(6)
            ],
            # conv5_x
            Conv2D(2048, kernel_size=(1, 1), strides=(2, 2)),
            [
                ResidualBlock(2048, 2048) for _ in range(3)
            ],
            # avg pooling
            GlobalAveragePooling2D(),
            Dense(1000, activation=tf.nn.relu),
            Dense(output_dim, activation=tf.nn.softmax)
        ]

    def call(self, x):
        for layer in self._layers:
            if isinstance(layer, list):
                for l in layer:
                    x = l(x)
            else:
                x = layer(x)
        return x


# build the model
model = ResNet50((28, 28, 1), 10)
model.build(input_shape=(None, 28, 28, 1))
model.summary()
# crossentropy loss function considered as there are two or more label classes.
# SparseCategoricalCrossentropy is more efficient when there are lots of categories.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()  # optimizing the model by using Adam algorithm

train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')

# train the model by creating a polymorphic callable function
@tf.function
def train_step(image, label):
    with tf.GradientTape() as tape:
        predictions = model(image)
        loss = loss_object(label, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # Processing aggregated gradients.

    train_loss(loss)
    train_accuracy(label, predictions)


# test the model by creating a polymorphic callable function
@tf.function
def test_step(image, label):
    predictions = model(image)
    loss = loss_object(label, predictions)

    test_loss(loss)
    test_accuracy(label, predictions)


num_epoch = 10
start_time = time.time()

train_accuracies = []
test_accuracies = []

for epoch in range(num_epoch):
    for image, label in dataset_train:
        for _image, _label in datagen.flow(image, label, batch_size=batch_size):
            train_step(_image, _label)
            break

    for test_image, test_label in dataset_test:
        test_step(test_image, test_label)

    train_accuracies.append(train_accuracy.result())
    test_accuracies.append(test_accuracy.result())

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, spent_time: {} min'
    spent_time = time.time() - start_time
    print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100, test_loss.result(),
                          test_accuracy.result() * 100, spent_time / 60))

# plotting the accuracy graphs
plt.plot(train_accuracies, label = 'Train Accuracy')
plt.plot(test_accuracies, linestyle = 'dashed', label = 'Test Accuracy')
plt.legend()
plt.show()