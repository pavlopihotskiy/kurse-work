import numpy as np
import tensorflow as tf
from app.config import MODEL_PATH, CLASSES


class TinyMLModel:
    def __init__(self):
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, features):
        x = np.expand_dims(features, axis=0)
        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )[0]

        class_id = int(np.argmax(output))
        return CLASSES[class_id], round(float(output[class_id]), 3)
