import io
import logging
import sys
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define a custom class to redirect output to the logging module
class LogStream(io.StringIO):
    def write(self, msg):
        # Log the message to the logging module
        logging.info(msg)

# Create instances of the custom class for stdout and stderr
stdout_logger = LogStream()
stderr_logger = LogStream()

# Redirect stdout and stderr to the custom class instances
sys.stdout = stdout_logger
sys.stderr = stderr_logger

# Configure logging to write to a file using UTF-8 encoding
logging.basicConfig(filename='predict_log.txt', level=logging.INFO, encoding='utf-8')

# Load Json
# Load the model architecture from the JSON file
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = tf.keras.models.model_from_json(loaded_model_json)

# Load the weights into the model
with open("model_weights.bin", "rb") as bin_file:
    for layer in loaded_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            # For BatchNormalization layers, load gamma and beta
            gamma_beta = np.fromfile(bin_file, dtype=np.float32, count=2 * layer.input[-1].shape[-1])
            gamma = gamma_beta[:layer.input[-1].shape[-1]]
            beta = gamma_beta[layer.input[-1].shape[-1]:]
            moving_mean = np.fromfile(bin_file, dtype=np.float32, count=layer.input[-1].shape[-1])
            moving_variance = np.fromfile(bin_file, dtype=np.float32, count=layer.input[-1].shape[-1])

            layer.set_weights([gamma, beta, moving_mean, moving_variance])
        else:
            # For other layers, load weights as usual
            layer_weights = [np.fromfile(bin_file, dtype=np.float32, count=np.prod(param.shape)).reshape(param.shape)
                             for param in layer.trainable_variables]
            layer.set_weights(layer_weights)


loaded_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                     loss=tf.keras.losses.BinaryCrossentropy(),
                     metrics=['accuracy'])

def predict_class(path):
    img = cv2.imread(path)
    RGBImg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    RGBImg= cv2.resize(RGBImg,(224,224))
    plt.imshow(RGBImg)
    image = np.array(RGBImg) / 255.0
    # new_model = tf.keras.models.load_model("64x3-CNN.model")
    predict=loaded_model.predict(np.array([image]))
    per=np.argmax(predict,axis=1)
    if per==1:
        print('Diabetic Retinopathy Detected')
    else:
        print('Diabetic Retinopathy Not Detected')
  

predict_class('F:\\SEM-8\\IoT Domain Analyst\\IoT Jcomp\\338_right.jpeg')