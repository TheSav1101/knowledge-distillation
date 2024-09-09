from __future__ import absolute_import
from tensorflow.keras.applications import MobileNetV3Small
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

import pickle
import math
import fractions
import os
import tensorflow as tf
import keras
import keras.layers as layers
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.config.run_functions_eagerly(True)


#####INSERT VARIABLES HERE#####
image_folder = "/home/savoia/H36M-Toolbox/images/"
joints_n = 17
teacher_path = "/home/savoia/metrabs/models/heatmaps_metrabs"

######DISTILLER#########

class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3, distillation_alpha=0.5):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.distillation_alpha = distillation_alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data
        teacher_predictions, teacher_heatmaps = self.teacher.detect_poses_batched(x, skeleton='h36m_17', default_fov_degrees=55, suppress_implausible_poses=False)
        with tf.GradientTape() as tape:
            student_predictions, student_heatmaps = self.student(x, training=True)

            student_predictions = np.array(student_predictions)

            student_predictions_filtered = student_predictions[1:]
            student_predictions_filtered = student_predictions_filtered.reshape(-1)
            y_np = y.numpy().reshape(-1)

            student_loss = self.student_loss_fn(y_np, student_predictions_filtered)
            student_predictions = student_predictions.transpose(1, 0, 2)
            teacher_predictions = teacher_predictions.detach().numpy().reshape(32,-1)
            teacher_predictions = teacher_predictions[:, :4*joints_n]
            teacher_predictions = teacher_predictions.reshape(32,4,joints_n)
            print((teacher_predictions / self.temperature).shape)
            print((student_predictions / self.temperature).shape)
            distillation_loss = self.distillation_loss_fn(
                teacher_predictions / self.temperature,
                student_predictions / self.temperature
            ) * (self.temperature ** 2)

            teacher_heatmaps = teacher_heatmaps.detach().numpy()
            teacher_heatmaps = teacher_heatmaps.reshape(32, -1)
            print((teacher_heatmaps / self.temperature).shape)
            student_heatmaps = np.array(student_heatmaps)
            student_heatmaps = student_heatmaps.transpose(1, 0, 2, 3, 4)
            student_heatmaps = student_heatmaps.reshape(32, -1)
            print((student_heatmaps / self.temperature).shape)
            heatmaps_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_heatmaps / self.temperature, axis=1).numpy(),
                tf.nn.softmax(student_heatmaps / self.temperature, axis=1).numpy()
            ) * (self.temperature ** 2)

            loss = self.alpha * student_loss + (1 - self.alpha) * (distillation_loss*(self.distillation_alpha) + heatmaps_loss*(1 - self.distillation_alpha))

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def test_step(self, data):
        x, y = data
        teacher_predictions, teacher_heatmaps = self.teacher.detect_poses_batched(x, skeleton='h36m_17', default_fov_degrees=55, suppress_implausible_poses=False)
        student_predictions, student_heatmaps = self.student(x, training=True)
        student_predictions = np.array(student_predictions)
        student_predictions_filtered = student_predictions[1:]
        student_predictions_filtered = student_predictions_filtered.reshape(-1)
        y_np = y.numpy().reshape(-1)
        student_loss = self.student_loss_fn(y_np, student_predictions_filtered)
        student_predictions = student_predictions.transpose(1, 0, 2)
        teacher_predictions = teacher_predictions.detach().numpy().reshape(32,-1)
        teacher_predictions = teacher_predictions[:, :4*joints_n]
        teacher_predictions = teacher_predictions.reshape(32,4,joints_n)
        print((teacher_predictions / self.temperature).shape)
        print((stutdent_predictions / self.temperature).shape)
        distillaion_loss = self.distillation_loss_fn(
            teacher_predictions / self.temperature,
            student_predictions / self.temperature
        ) * (self.temperature ** 2)

        teacher_heatmaps = teacher_heatmaps.detach().numpy()
        teacher_heatmaps = teacher_heatmaps.reshape(32, -1)
        print((teacher_heatmaps / self.temperature).shape)
        student_heatmaps = np.array(student_heatmaps)
        student_heatmaps = student_heatmaps.transpose(1, 0, 2, 3, 4)
        student_heatmaps = student_heatmaps.reshape(32, -1)
        print((student_heatmaps / self.temperature).shape)
        heatmaps_loss = self.distillation_loss_fn(
            tf.nn.softmax(teacher_heatmaps / self.temperature, axis=1).numpy(),
            tf.nn.softmax(student_heatmaps / self.temperature, axis=1).numpy()
        ) * (self.temperature ** 2)

        loss = self.alpha * student_loss + (1 - self.alpha) * (distillation_loss*(self.distillation_alpha) + heatmaps_loss*(1 - self.distillation_alpha))

        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def call(self, inputs):
        return self.student(inputs)


#######SETUP TEACHER########
teacher = tf.saved_model.load(teacher_path)



#######SETUP STUDNET########
mobilenet_base = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
mobilenet_layers = mobilenet_base.layers
selected_layers = mobilenet_layers #[:115] #Cut as in paper best = 86
student_backbone = keras.Model(mobilenet_base.input, selected_layers[-1].output, name="student_backbone")
for layer in student_backbone.layers:
    layer.trainable = False


#block 13_a
top = layers.Conv2D(368, 1, activation="relu")(student_backbone.output)
top = layers.DepthwiseConv2D(3, activation="relu", padding="same")(top)
top = layers.Conv2D(256, 1)(top) #### NO RELU
bot = layers.Conv2D(256, 1)(student_backbone.output)
block_13_a_output = layers.add([top, bot])

#block 13_b
top = layers.Conv2D(192, 1, activation="relu")(block_13_a_output)
top = layers.DepthwiseConv2D( 3, activation="relu", padding="same")(top)
block_13_b_output = layers.Conv2D(192, 1, activation="relu")(top)

top = layers.UpSampling2D(interpolation="bilinear")(block_13_b_output)
top = layers.Conv2D(128, 3, activation="relu")(top)

mid = layers.UpSampling2D(interpolation="bilinear")(block_13_b_output)
mid = layers.Conv2D(3 * joints_n, 3)(mid)

class JointLength(layers.Layer):
    def call(self, x):
        delta_x, delta_y, delta_z = tf.split(x, num_or_size_splits=3, axis=3)
        return tf.abs(delta_x) + tf.abs(delta_y) + tf.abs(delta_z)

bl = JointLength()(mid)

concat = layers.Concatenate()([top, mid, bl])
fin = layers.Conv2D(128, 1, activation="relu")(concat)
fin = layers.DepthwiseConv2D(3, activation="relu", padding="same")(fin)
fin = layers.Conv2D(4 * joints_n, 1)(fin)

class HeatmapLayer(layers.Layer):
    def call(self, x):
      H,X,Y,Z = tf.split(x,num_or_size_splits=4, axis=3)
      return [H,X,Y,Z]

class OutputLayer(layers.Layer):
    def call(self, x):
      outs = []
      for y in x:
        out = layers.Flatten()(y)
        out = layers.Dense(joints_n)(out)
        outs.append(out)
      return outs

st_heats = HeatmapLayer()(fin)
st_output = OutputLayer()(st_heats)
student = keras.Model(student_backbone.input, [st_output, st_heats], name = "student")
student.summary()
student_scratch = keras.models.clone_model(student)


import matplotlib.pyplot as plt
def plot_history(history):
  print(history.history.keys())
  #  "Accuracy"
  plt.plot(history.history['sparse_categorical_accuracy'])
  plt.plot(history.history['val_sparse_categorical_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()
  # "Loss"
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()


##################################################
#################DATA SETUP#######################



with open('h36m_train.pkl', 'rb') as f:
    dataset_train = pickle.load(f)

images = []
labels = []

i=0

for bb in dataset_train:
    img = cv2.imread(image_folder + bb['image'])
    box = bb['box']
    boxed_img = img[int(box[0]):int(box[2]), int(box[1]):int(box[3])]
    x = cv2.resize(boxed_img, (224, 224))
    y = bb['joints_3d']

    images.append(x)
    labels.append(y)
    i+=1
    if(i >=100):
        break




images = np.array(images)
labels = np.array(labels)

x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

##################################################
#####################TRAINING#####################

# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.MeanSquaredError(), #keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.5,
    temperature=3,
)

# Distill teacher to student
history = distiller.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val), shuffle=True)
plot_history(history)


# Train student as doen usually
student_scratch.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train and evaluate student trained from scratch.
history = student_scratch.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val), shuffle=True)
plot_history(history)