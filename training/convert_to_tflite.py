import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("yes_no_model")
tflite = converter.convert()

with open("yes_no_model.tflite", "wb") as f:
    f.write(tflite)
