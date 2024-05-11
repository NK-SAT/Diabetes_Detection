import json
import numpy as np
from azureml.core.model import Model

# Load the model architecture from the registered model
def init():
    model_path = Model.get_model_path("diabetes_prediction_model")

    # Load the JSON file containing the model architecture
    with open(os.path.join(model_path, "model.json"), "r") as json_file:
        loaded_model_json = json_file.read()

    # Reconstruct the Keras model from the JSON data
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)

    # Load the weights into the model
    with open(os.path.join(model_path, "model_weights.bin"), "rb") as bin_file:
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

# Call the function to load the model
def run(img):
    #img = cv2.imread(path)
    RGBImg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    RGBImg= cv2.resize(RGBImg,(224,224))
    plt.imshow(RGBImg)
    image = np.array(RGBImg) / 255.0
    # new_model = tf.keras.models.load_model("64x3-CNN.model")
    predict=loaded_model.predict(np.array([image]))
    per=np.argmax(predict,axis=1)
    if per==1:
        return json.dumps({"result": "Diabetic Retinopathy Detected"})
    else:
        return json.dumps({"result": "Diabetic Retinopathy Not Detected"})