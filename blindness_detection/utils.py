import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split

PATH_DF_CSV = ''
PATH_OF_IMAGE = ''

def dataframe():
    df = pd.read_csv(PATH_DF_CSV)
    df.rename(columns={'image': 'id_code', 'level': 'diagnosis'}, inplace=True)
    return df

def create_path():
    df = dataframe()
    x = df['id_code']
    y = df['diagnosis']
    paths = []
    for i in x:
        path = PATH_OF_IMAGE+f'{i}.jpeg'
        paths.append(path)
    return paths

def load_data(subset=None):
    paths = create_path()
    imgs = []
    for path in paths[:subset]:
        img = cv2.imread(path)
        imgs.append(np.array(img))
    X = np.array(imgs)
    return X

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

def split(x, y, test_size=0.15, random_state=8):

    print("retour",test_size, random_state)
    yy = y.copy()

    #split train set
    train_x, valid_x, train_y, valid_y = train_test_split(x, yy, test_size=test_size,
                                                      stratify=yy, random_state=random_state)
    return train_x, valid_x, train_y, valid_y
