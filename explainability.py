import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg') # CRITICAL: Prevents matplotlib from trying to open local GUI windows on your server
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap

def run_complete_xai_pipeline(img_path, model, last_conv_layer_name):
    """
    LIME Saves the visual explanations to the static folder and returns their filenames.
    """
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    IMAGE_SIZE = 256  

    # =================------------------------
    # 1️⃣ LOAD AND PREPROCESS IMAGE
    # =================------------------------
    img = tf.keras.utils.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array_norm = img_array / 255.0  
    img_array_batch = tf.expand_dims(img_array_norm, axis=0)

    # LIME SPECIFIC LOGIC
    def predict_fn(images):
        return model(images).numpy()

    explainer = lime_image.LimeImageExplainer()
    lime_exp = explainer.explain_instance(
        img_array_norm, 
        predict_fn, 
        top_labels=1, 
        hide_color=0, 
        num_samples=100 
    )
    
    lime_img, lime_mask = lime_exp.get_image_and_mask(
        lime_exp.top_labels[0], 
        positive_only=True, 
        num_features=5, 
        hide_rest=False
    )
    
    # Render boundaries and scale back up to save cleanly
    lime_bounded = mark_boundaries(lime_img, lime_mask)
    lime_filename = f"xai_lime_{base_name}.png"
    plt.imsave(os.path.join('static', lime_filename), lime_bounded)

    return  lime_filename