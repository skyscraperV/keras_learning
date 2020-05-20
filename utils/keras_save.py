import keras
import os
import tensorflow as tf

tf.executing_eagerly()

def export_savedmodel_pb(model,model_path="model"):
    tf.keras.models.save_model(model, model_path)
    print("save model pb success ...")
