import numpy as np
import pandas as pd
import cv2
from data_loading.py import load_data

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
CSV_PATH = os.path.join(ROOT_DIR, 'raw_data', 'kaggle_clean_dataset', 'train.csv')
TRAIN_IMAGES_PATH = os.path.join(ROOT_DIR, 'raw_data', 'kaggle_clean_dataset', 'train_images/')

def preprocessing_1_autocropping(sigmaX=10):
    """
    Create circular crop around image centre and applies Ben Graham's color
    """

    X = load_data()

    images = []

    for image in X:
        img = first_cropping(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, depth = img.shape

        x = int(width/2)
        y = int(height/2)
        r = np.amin((x,y))

        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
        img = cv2.bitwise_and(img, img, mask=circle_img)
        img = first_cropping(img)
        img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)

        images.append(np.array(img))

    preprocessed_X_autocropping = np.array(images)
    return preprocessed_X_autocropping

def first_cropping(img,tol=7):
    '''
    tol is tolerance, img is image data, creates first cropping
    '''

    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img

def preprocessing_2_same_size():
    X = load_data()
    images = []
    for image in X:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = first_cropping(image)
        image = cv2.resize(image, (512, 512))
        image=cv2.addWeighted(image,4, cv2.GaussianBlur(image , (0,0) , 10) ,-4 ,128)
        images.append(np.array(image))

    preprocessed_X_same_size = np.array(images)

    return preprocessed_X_same_size
