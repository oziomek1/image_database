import numpy as np
import cv2
import os


def cropper():
    """Crop images according to set values for x & y, width & height"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(directory):
                frame = cv2.imread(os.path.join(directory, filename), 1)
                y = 500
                x = 650
                size = 300
                cropped_image = frame[y:(y + size), x:(x + size)]
                base_filename = os.path.splitext(filename)[0]
                temp_name = str(base_filename) + '_cropped.jpg'
                print(filename, temp_name)
                cv2.imwrite(os.path.join(directory, temp_name), cropped_image)
                # os.rename(os.path.join(directory, filename), os.path.join(directory, temp_name))


def dropper():
    """Drop images using it name"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(directory):
                if filename.endswith('scaled2.jpg'):
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


def scaler2():
    """Resize image to appropriate size"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(directory):
                frame = cv2.imread(os.path.join(directory, filename), 1)
                scaled_frame = cv2.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
                base_filename = os.path.splitext(filename)[0]
                cv2.imwrite(os.path.join(directory, base_filename + '_scaled2.jpg'), scaled_frame)


windowname = 'Cropper'
cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
image = './1/1.jpg'

# dropper()
# cropper()
# rotator()
# grayer()
# scaler2()

originalFrame = cv2.imread(image, 1)
if originalFrame is None:
    raise Exception('cannot read')

# originalFrame = cv2.resize(originalFrame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

# Change originalFrame to grayscale
# processedFrame = cv2.cvtColor(originalFrame, cv2.COLOR_BGR2GRAY)

# Display the results
y = 500
x = 650
size = 360
cropped_image = originalFrame[y:(y+size), x:(x+size)]
cv2.imshow(windowname, cropped_image)
cv2.waitKey(0)
# Release capturing video
cv2.destroyAllWindows()
