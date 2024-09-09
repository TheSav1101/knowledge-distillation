from __future__ import absolute_import
from tensorflow.keras.applications import MobileNetV3Small
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

import pickle
import os
import tensorflow as tf
import keras
import keras.layers as layers
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

tf.config.run_functions_eagerly(True)

#####INSERT VARIABLES HERE#####
colab_folder = "./"
image_folder = "/home/savoia/H36M-Toolbox/images/"
teacher_name = "models/SelecSLS60_statedict_better"
teacher_ext = ".pth"
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
        scaling_factor=0.01

        x=tf.convert_to_tensor(x, dtype=tf.uint8)
        y=tf.convert_to_tensor(y)
        N=x.shape[0]
        boxes_inner=tf.constant([0,0,244,244], dtype=tf.float32)
        boxes= tf.RaggedTensor.from_row_lengths(
            values=tf.tile(tf.expand_dims(boxes_inner, axis=0), [N,1]),
            row_lengths=[1]*N
        )


        teacher_output = self.teacher.estimate_poses_batched(x, boxes, skeleton='h36m_17', default_fov_degrees=55) #, max_detections=1, suppress_implausible_poses=False)
        teacher_predictions = teacher_output['poses3d'].to_tensor()
        teacher_predictions = tf.squeeze(teacher_predictions, 1)
        teacher_heatmaps = teacher_output['logits2d'].to_tensor()
        teacher_heatmaps = tf.squeeze(teacher_heatmaps, 1)

        with tf.GradientTape() as tape:
            student_predictions, student_heatmaps = self.student(x, training=True)
            student_predictions = tf.transpose(student_predictions, (1,2,0))
            student_heatmaps = tf.transpose(student_heatmaps, (0,3,1,2))


            #print("teacher heatmaps shape: " + str(teacher_heatmaps.shape))
            #print("student heatmaps shape: " + str(student_heatmaps.shape))
            #print("teacher shape:          " + str(teacher_predictions.shape))
            #print("student shape:          " + str(student_predictions.shape))

            #print("Labels shape:           " + str(y.shape))
            student_loss = self.student_loss_fn(y, student_predictions)

            #print("student loss:           " + str(student_loss))

            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1)
            ) * (self.temperature ** 2)
            #print("distillation loss:      " + str(distillation_loss))

            heatmaps_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_heatmaps / self.temperature, axis=1),
                tf.nn.softmax(student_heatmaps / self.temperature, axis=1)
            ) * (self.temperature ** 2)
            #print("heatmap loss:           " + str(heatmaps_loss))

            loss = self.alpha * student_loss + (1 - self.alpha) * (distillation_loss*(self.distillation_alpha) + heatmaps_loss*(1 - self.distillation_alpha))*scaling_factor
            #print("total loss:             " + str(loss))
            if tf.reduce_any(tf.math.is_nan(loss)):
                print("NaN loss, skipping batch...")
                return None

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        #print("gradients: ")
        #for g in gradients:
        #    print(g)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        #print("student shape: " + str(student_predictions.shape))
        #print("Labels shape:  " + str(y.shape))
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def test_step(self, data):
        x, y = data
        x=tf.convert_to_tensor(x, dtype=tf.uint8)
        y=tf.convert_to_tensor(y)
        N=x.shape[0]
        boxes_inner=tf.constant([0,0,244,244], dtype=tf.float32)
        boxes= tf.RaggedTensor.from_row_lengths(
            values=tf.tile(tf.expand_dims(boxes_inner, axis=0), [N,1]),
            row_lengths=[1]*N
        )

        teacher_output = self.teacher.estimate_poses_batched(x, boxes, skeleton='h36m_17', default_fov_degrees=55) #, max_detections=1, suppress_implausible_poses=False)
        teacher_predictions = teacher_output['poses3d'].to_tensor()
        teacher_predictions = tf.squeeze(teacher_predictions, 1)
        teacher_heatmaps = teacher_output['logits2d'].to_tensor()
        teacher_heatmaps = tf.squeeze(teacher_heatmaps, 1)

        student_predictions, student_heatmaps = self.student(x, training=False)
        student_predictions = tf.transpose(student_predictions, (1,2,0))
        student_heatmaps = tf.transpose(student_heatmaps, (0,3,1,2))

        student_loss = self.student_loss_fn(y, student_predictions)
        distillation_loss = self.distillation_loss_fn(
            tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
            tf.nn.softmax(student_predictions / self.temperature, axis=1)
        ) * (self.temperature ** 2)

        heatmaps_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_heatmaps / self.temperature, axis=1),
                tf.nn.softmax(student_heatmaps / self.temperature, axis=1)
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
#fin = layers.Conv2D(3 * joints_n, 1)(fin)

#class HeatmapLayer(layers.Layer):
#    def call(self, x):
#      X,Y,Z = tf.split(x,num_or_size_splits=3, axis=3)
#      return [X,Y,Z]
#class OutputLayer(layers.Layer):
#    def call(self, x):
#      outs = []
#      for y in x:
#        out = layers.Flatten()(y)
 #       out = layers.Dense(joints_n)(out)
  #      outs.append(out)
  #    return outs

class OutputLayer(layers.Layer):
    def call(self, x):
      outs = []
      for y in range(3):
        out = layers.Flatten()(x)
        out = layers.Dense(joints_n)(out)
        outs.append(out)
      return outs

st_heats = layers.Conv2D(joints_n, 1)(fin)
st_output = OutputLayer()(st_heats)
student = keras.Model(student_backbone.input, [st_output, st_heats], name = "student")
student.summary()
student_scratch = keras.models.clone_model(student)


import matplotlib.pyplot as plt
def plot_history(history):
  print(history.history.keys())
  #  "Accuracy"
  plt.plot(history.history['mean_absolute_error'])
  plt.plot(history.history['val_mean_absolute_error'])
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

unused, dataset_small = train_test_split(dataset_train, test_size=0.15, random_state=42)
train_dataset, val_dataset = train_test_split(dataset_small, test_size=0.2, random_state=42)

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, image_folder, batch_size=64, image_size=(224, 224), shuffle=True):
        self.data = data
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self.data[k] for k in indexes]
        X, y = self.__data_generation(batch_data)

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_data):

        X = np.empty((len(batch_data), *self.image_size, 3), dtype=np.uint8)
        y = np.empty((len(batch_data), joints_n, 3), dtype=np.float32)

        for i, bb in enumerate(batch_data):
            try:
                img_path = self.image_folder + bb['image']
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Image at {img_path} could not be read.")
                    continue

                box = bb['box']
                boxed_img = img[int(box[0]):int(box[2]), int(box[1]):int(box[3])]

                if boxed_img.size == 0:
                    print(f"Image at {img_path} has size 0.")
                    continue

                x = cv2.resize(boxed_img, self.image_size)
                y[i] = bb['joints_3d']
                X[i,] = x.astype(np.uint8)

            except Exception as e:
                print(f"An error occurred while loading image: {e}")

        return X, y

train_generator = DataGenerator(train_dataset, image_folder, batch_size=32, image_size=(224, 224), shuffle=True)
val_generator = DataGenerator(val_dataset, image_folder, batch_size=32, image_size=(224, 224), shuffle=False)

#images = []
#labels = []

#i=0

#for bb in tqdm(dataset_train, desc="Img loading"):
#    try:
#        img = cv2.imread(image_folder + bb['image'])
#        box = bb['box']
#        boxed_img = img[int(box[0]):int(box[2]), int(box[1]):int(box[3])]
#        if boxed_img.size != 0:
#	        x = cv2.resize(boxed_img, (224, 224))
#	        y = bb['joints_3d']
#	        images.append(x)
	#        labels.append(y)
 #   except:
#        print("one image was not read correctly")

#images = np.array(images)
#labels = np.array(labels)

#x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

##################################################
#####################TRAINING#####################

# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.MeanAbsoluteError()],
    student_loss_fn=keras.losses.Huber(),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.6,
    temperature=3,
)

# Distill teacher to student
history = distiller.fit(train_generator, epochs=20, validation_data=val_generator, shuffle=True)
plot_history(history)

distiller.save('./metrabs_studnet_1')
