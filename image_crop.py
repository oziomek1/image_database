import numpy as np
import cv2
import os
import shutil
import random


def copier():
    """Copy the file from one directory to another"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir('original/.'):
        if directory in dir_list:
            print(directory)
            for filename in os.listdir('original/' + directory):
                print(filename)
                shutil.copyfile('/home/oziomek/licencjat/kostki/zdjecia/original/' + directory + '/' + filename,
                                '/home/oziomek/licencjat/kostki/zdjecia/' + directory + '/' + filename)


def create_dirs():
    """Create directories"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        if directory in dir_list:
            try:
                os.makedirs(directory, exist_ok=False)
            except OSError:
                if not os.path.isdir(directory):
                    raise


def name_changer():
    """Change the name of each photo"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        if directory in dir_list:
            for filename in os.listdir(directory):
                if filename.endswith('scaled.jpg'):
                    new_filename = filename.split('_')
                    os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename[0] + '.jpg'))


def cropper():
    """Crop images according to set values for x & y, width & height"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(directory):
                frame = cv2.imread(os.path.join(directory, filename), 1)
                y = 80 # 500*0.1866
                x = 118 # 650*0.1866
                size = 64 # 300*0.1866
                cropped_image = frame[y:(y + size), x:(x + size)]
                base_filename = os.path.splitext(filename)[0]
                temp_name = str(base_filename) + '_cropped.jpg'
                print(filename, temp_name)
                cv2.imwrite(os.path.join(directory, temp_name), cropped_image)
                # os.rename(os.path.join(directory, filename), os.path.join(directory, temp_name))


def remover():
    """Remove images using it name"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(directory):
                if filename.endswith('cropped.jpg'):
                    os.remove(os.path.join(directory, filename))


def rotator():
    """Rotates images by 90*, 180* and 270*"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(directory):
                if filename.endswith('cropped.jpg'):
                    frame = cv2.imread(os.path.join(directory, filename), 1)
                    frame = cv2.transpose(frame)
                    frame_CV = cv2.flip(frame, 1)
                    frame_CCV = cv2.flip(frame, 2)
                    frame_180 = cv2.flip(frame, -1)
                    base_filename = os.path.splitext(filename)[0]
                    print(base_filename)
                    # cv2.imwrite(os.path.join(directory, base_filename + '_CV.jpg'), frame_CV)
                    # cv2.imwrite(os.path.join(directory, base_filename + '_CCV.jpg'), frame_CCV)
                    cv2.imwrite(os.path.join(directory, base_filename + '_180.jpg'), frame_180)


def grayer():
    """Convert images to grayscale using BGR2GRAY"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(directory):
                frame = cv2.imread(os.path.join(directory, filename), 1)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                base_filename = os.path.splitext(filename)[0]
                cv2.imwrite(os.path.join(directory, base_filename + '_gray.jpg'), gray_frame)


def scaler():
    """Resize image to appropriate size"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(directory):
                frame = cv2.imread(os.path.join(directory, filename), 1)
                scaled_frame = cv2.resize(frame, None, fx=0.1866, fy=0.1866, interpolation=cv2.INTER_CUBIC)
                base_filename = os.path.splitext(filename)[0]
                cv2.imwrite(os.path.join(directory, base_filename + '_scaled.jpg'), scaled_frame)


def rotator_angle():
    """Rotate images according to angle"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(directory):
                print('\t' + filename)
                frame = cv2.imread(os.path.join(directory, filename), 1)
                base_filename = os.path.splitext(filename)[0]
                frame_center = tuple(np.array(frame.shape[1::-1]) /2)
                for i in range(0, 360, 5):
                    if i is 0: continue
                    matrix = cv2.getRotationMatrix2D(frame_center, i, 1)
                    result_image = cv2.warpAffine(frame, matrix, frame.shape[1::-1], flags=cv2.INTER_LINEAR)
                    cv2.imwrite(os.path.join(directory, base_filename + '_' + str(i) + '_rotation' + '.jpg'), result_image)


def remove_rotated():
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(directory):
                if filename.endswith('_rotation.jpg'):
                    os.remove(os.path.join(directory, filename))


windowname = 'Cropper'
cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
image = './1/1.jpg'

# create_dirs()
# copier()
# remover()
# cropper()
# rotator()
# grayer()
# scaler()
# name_changer()
rotator_angle()
# remove_rotated()

# originalFrame = cv2.imread(image, 1)
# if originalFrame is None:
#     raise Exception('cannot read')

# originalFrame = cv2.resize(originalFrame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

# Change originalFrame to grayscale
# processedFrame = cv2.cvtColor(originalFrame, cv2.COLOR_BGR2GRAY)

# Display the results
# y = 500
# x = 650
# size = 360
# cropped_image = originalFrame[y:(y + size), x:(x + size)]
# cv2.imshow(windowname, cropped_image)
# cv2.waitKey(0)
# # Release capturing video
# cv2.destroyAllWindows()
