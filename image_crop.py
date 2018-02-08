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
    for directory in os.listdir("original/"):
        if directory in dir_list:
            try:
                os.makedirs(directory, exist_ok=False)
            except OSError:
                if not os.path.isdir(directory):
                    raise


def name_changer(name):
    """Change the name of each photo"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        if directory in dir_list:
            for filename in os.listdir(directory):
                if filename.endswith(name):
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
                # base_filename = os.path.splitext(filename)[0]
                # temp_name = str(base_filename) + '.jpg'
                temp_name = str(filename)
                os.remove(os.path.join(directory, filename))
                cv2.imwrite(os.path.join(directory, temp_name), cropped_image)
                # os.rename(os.path.join(directory, filename), os.path.join(directory, temp_name))


def change_perspective1():
    """Change perspective getPerspectiveTransform and using warpPerspective"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(directory):
                frame = cv2.imread(os.path.join(directory, filename), 1)
                base_filename = os.path.splitext(filename)[0]
                if base_filename.endswith('_1_ppct.jpg'):
                    continue
                width = 299
                height = 224
                size = 68

                pts1 = np.float32([[width-2, height], [width + size, height-1], [width + size, height + size], [width, height + size]])
                pts2 = np.float32([[width, height], [width + size, height], [width + size, height + size], [width, height + size]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(frame, M, (width, height))
                temp_name = str(base_filename) + '_1_ppct.jpg'
                cv2.imwrite(os.path.join(directory, temp_name), dst)


def change_perspective2():
    """Change perspective getPerspectiveTransform and using warpPerspective"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(directory):
                frame = cv2.imread(os.path.join(directory, filename), 1)
                base_filename = os.path.splitext(filename)[0]
                if base_filename.endswith('_ppct'):
                    continue
                width = 299
                height = 224
                size = 68

                pts1 = np.float32([[width+2, height], [width + size, height], [width + size-2, height + size+1], [width+1.5, height + size+1]])
                pts2 = np.float32([[width, height], [width + size, height], [width + size, height + size], [width, height + size]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(frame, M, (width, height))
                temp_name = str(base_filename) + '_2_ppct.jpg'
                cv2.imwrite(os.path.join(directory, temp_name), dst)


def change_perspective3():
    """Change perspective getPerspectiveTransform and using warpPerspective"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(directory):
                frame = cv2.imread(os.path.join(directory, filename), 1)
                base_filename = os.path.splitext(filename)[0]
                if base_filename.endswith('_ppct'):
                    continue
                width = 299
                height = 224
                size = 68

                pts1 = np.float32([[width, height], [width + size, height], [width + size, height + size], [width-2, height + size+2]])
                pts2 = np.float32([[width, height], [width + size, height], [width + size, height + size], [width, height + size]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(frame, M, (width, height))
                temp_name = str(base_filename) + '_3_ppct.jpg'
                cv2.imwrite(os.path.join(directory, temp_name), dst)


def change_perspective4():
    """Change perspective getPerspectiveTransform and using warpPerspective"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(directory):
                frame = cv2.imread(os.path.join(directory, filename), 1)
                base_filename = os.path.splitext(filename)[0]
                if base_filename.endswith('_ppct'):
                    continue
                width = 299
                height = 224
                size = 68

                pts1 = np.float32([[width, height+2], [width + size, height+4], [width + size+2, height + size], [width, height + size]])
                pts2 = np.float32([[width, height], [width + size, height], [width + size, height + size], [width, height + size]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(frame, M, (width, height))
                temp_name = str(base_filename) + '_4_ppct.jpg'
                cv2.imwrite(os.path.join(directory, temp_name), dst)


def change_perspective5():
    """Change perspective getPerspectiveTransform and using warpPerspective"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(directory):
                frame = cv2.imread(os.path.join(directory, filename), 1)
                base_filename = os.path.splitext(filename)[0]
                if base_filename.endswith('_ppct'):
                    continue
                width = 299
                height = 224
                size = 68

                pts1 = np.float32([[width+2, height-2], [width + size, height], [width + size+2, height + size], [width, height + size]])
                pts2 = np.float32([[width, height], [width + size, height], [width + size, height + size], [width, height + size]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(frame, M, (width, height))
                temp_name = str(base_filename) + '_5_ppct.jpg'
                cv2.imwrite(os.path.join(directory, temp_name), dst)


def change_perspective6():
    """Change perspective getPerspectiveTransform and using warpPerspective"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(directory):
                frame = cv2.imread(os.path.join(directory, filename), 1)
                base_filename = os.path.splitext(filename)[0]
                if base_filename.endswith('_ppct'):
                    continue
                width = 299
                height = 224
                size = 68

                pts1 = np.float32([[width, height+1], [width + size, height-2], [width + size, height + size-2], [width+1, height + size]])
                pts2 = np.float32([[width, height], [width + size, height], [width + size, height + size], [width, height + size]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(frame, M, (width, height))
                temp_name = str(base_filename) + '_6_ppct.jpg'
                cv2.imwrite(os.path.join(directory, temp_name), dst)


def remover(name):
    """Remove images using it name"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(directory):
                if not filename.endswith(name):
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
                cv2.imwrite(os.path.join(directory, filename), gray_frame)


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
                frame_center = tuple(np.array(frame.shape[1::-1])/2)
                for i in range(0, 360, 5):
                    if i is 0:
                        continue
                    matrix = cv2.getRotationMatrix2D(frame_center, i, 1)
                    result_image = cv2.warpAffine(frame, matrix, frame.shape[1::-1], flags=cv2.INTER_LINEAR)
                    cv2.imwrite(os.path.join(directory, base_filename + '_' + str(i) + '_rot' + '.jpg'), result_image)


def remove_rotated():
    """Remove rotated images"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir("."):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(directory):
                if filename.endswith('_rot.jpg'):
                    os.remove(os.path.join(directory, filename))


# create_dirs()
# remover(".")
# copier()
# remover('_ppct.jpg')
# cropper()
# rotator()
# grayer()
# scaler()
# remover('_scaled.jpg')
# name_changer('_scaled.jpg')
# rotator_angle()
# remove_rotated()
change_perspective1()
change_perspective2()
change_perspective3()
change_perspective4()
change_perspective5()
change_perspective6()

windowname = 'Image Processing'
cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
image = 'original/2/1.jpg'

originalFrame = cv2.imread(image, 1)
if originalFrame is None:
    raise Exception('cannot read')

originalFrame = cv2.resize(originalFrame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

# Change originalFrame to grayscale
processedFrame = cv2.cvtColor(originalFrame, cv2.COLOR_BGR2GRAY)



# morphological transformations
# kernel = np.ones((2, 2), np.uint8)

# erosion
# erosion = cv2.erode(processedFrame, kernel, iterations=1)
# dilation
# dilation = cv2.dilate(processedFrame, kernel, iterations=1)
# opening
# opening = cv2.morphologyEx(processedFrame, cv2.MORPH_OPEN, kernel)

# Display the results
y = 500
x = 650
size = 360

y /= 4
x /= 4
size /= 4
x, y, size = int(x), int(y), int(size)


pts1 = np.float32([[y, x], [y+size, x], [y+size, x+size], [y, x+size]])
pts11 = np.float32([[y, x], [y+size+10, x], [y+size, x+size], [y, x+size]])
pts2 = np.float32([[y-20, x-20], [y+size+20, x+20], [y+size+20, x+size-20], [y-20, x+size+20]])


M = cv2.getPerspectiveTransform(pts1, pts11)
dst = cv2.warpPerspective(processedFrame, M, (400, 300))
cropped_image = dst[y:(y + size), x:(x + size)]
# cv2.imshow(windowname, cropped_image)
cv2.imshow(windowname, dst)
# cv2.waitKey(0)
# Release capturing video
cv2.destroyAllWindows()
