import numpy as np
import pandas as pd
import cv2

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
#import ipdb; ipdb.set_trace()

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
CSV_PATH = os.path.join(ROOT_DIR, 'raw_data', 'kaggle_clean_dataset', 'train.csv')
TRAIN_IMAGES_PATH = os.path.join(ROOT_DIR, 'raw_data', 'kaggle_clean_dataset', 'train_images/')

def dataframe():
    df = pd.read_csv(CSV_PATH)
    df.rename(columns={'image': 'id_code', 'level': 'diagnosis'}, inplace=True)
    return df

def create_paths():
    df = dataframe()
    x = df['id_code']
    paths = []
    for i in x:
        path = TRAIN_IMAGES_PATH+f'{i}'
        paths.append(path)
    return paths

def load_X(subset=None):
    paths = create_paths()
    imgs = []
    for path in paths[:subset]:
        img = cv2.imread(path)
        imgs.append(np.array(img))
    X = np.array(imgs)
    return X

def load_y():
    df = dataframe()
    y = df['diagnosis']
    y = to_categorical(y, num_classes = 5)
    return y

def split(x, y, test_size=0.15, random_state=8):
    x = load_X()
    y = load_y()
    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=test_size,
                                                      stratify=y, random_state=random_state)
    return train_x, valid_x, train_y, valid_y
