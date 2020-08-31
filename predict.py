from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import argparse, sys
import json
import os


img_size = 224
def process_image(img):
    img = tf.image.convert_image_dtype(img, dtype = tf.float32)
    img = tf.image.resize(img, (img_size, img_size))
    return img.numpy()


def predict(img_path, mdl, top_k):
    img = np.asarray(Image.open(img_path))
    img = process_image(img)
    img = np.expand_dims(img, axis = 0)
    ret = mdl.predict(img)
    df = pd.DataFrame(data = ret[0], index = list(range(len(ret[0]))), columns = ['prob'])
    df = df.sort_values(by = ['prob'], ascending = False)
    df.index.name = 'class'
    df.reset_index(inplace=True)
    cls = df['class'].to_list()
    cls = [x + 1 for x in cls]
    prbs = df['prob'].to_list()
    return prbs[:top_k], cls[:top_k]


def main():
    #Get filepath and dir
    dir = os.path.dirname(__file__)

    #Get arguments/options
    parser = argparse.ArgumentParser(description = 'Get class probabilities for an image based off a machine learning model')
    parser.add_argument('image_path', metavar = 'I', help = 'The path to the .jpg file that the model will use')
    parser.add_argument('model_path', metavar = 'M', help = 'The path to the .h5 model file that the program will use')
    parser.add_argument('--top_k', action = 'store', dest = 'top_k', type = int, default = 5)
    parser.add_argument('--category_names', action = 'store', dest = 'category_names', default = 'label_map.json')
    results = parser.parse_args()
    image_path = results.image_path
    if not os.path.isfile(image_path):
        print(f"{image_path} is not a valid file")
    model_path = results.model_path
    if not os.path.isfile(model_path) or '.h5' not in model_path:
        print(f"{model_path} is not a valid .h5 model file")
    top_k = results.top_k
    json_path = results.category_names
    if not os.path.isfile(json_path) or '.json' not in model_path:
        print(f"{json_path} is not a valid .json JSON file")

    with open(json_path, 'r') as fin:
        class_names = json.load(fin)

    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})

    probs, classes = predict(image_path, model, 5)
    classes = [class_names[str(x)] for x in classes]

    print('\nResults:')
    for p, c in zip(probs[:top_k], classes[:top_k]):
        print(f"Class {c} has probability {p}")


if __name__ == '__main__':
    main()
