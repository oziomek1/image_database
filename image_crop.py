import numpy as np
import cv2
import os
import shutil
import random
import time


"""
THIS FILE I MADE PURELY FOR SIMPLE YET TIME EFFICIENT OPERATIONS ON PHOTOS

THE CODE STYLE OF THIS FILE IS TERRIBLE HOWEVER ITS PUBLISHED ONLY TO PROVIDE
THIS CODE FOR VARIOUS COMPUTERS WITHOUT NEED TO COPYING IT FROM ONE TO ANOTHER MANUALLY 

"""


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


def create_empty_dirs(path):
    try:
        os.mkdir(path)
    except OSError:
        if not os.path.isdir(path):
            raise
    dirs = list(map(str, range(1, 7)))
    for directory in dirs:
        try:
            print(path + directory)
            os.mkdir(path + directory)
        except OSError:
            if not os.path.isdir(path + directory):
                raise


def name_simpler(path):
    """Change the name of each photo"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir(path):
        if directory in dir_list:
            i = 1
            print(path + directory)
            for filename in os.listdir(path + directory):
                # if filename.startswith('2018'):
                os.rename(os.path.join(path + directory, filename), os.path.join(path + directory, '2018-03-25-' + str(int(round(time.time() * 1000))) + '.jpg'))
                i += 1
                time.sleep(0.05)



def name_changer(path, name):
    """Change the name of each photo"""
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir(path):
        if directory in dir_list:
            print(path + directory)
            for filename in os.listdir(path + directory):
                if filename.endswith(name):
                    new_filename = filename.split('_')
                    # print(filename)
                    os.rename(os.path.join(path + directory, filename), os.path.join(path + directory, new_filename[0] + '.jpg'))


def change_perspective1(path, factor):
    """Change perspective getPerspectiveTransform and using warpPerspective"""
    print('Perspective 1')
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir(path):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(path + directory):
                frame = cv2.imread(os.path.join(path + directory, filename), 1)
                base_filename = os.path.splitext(filename)[0]
                if base_filename.endswith('_1_ppct.jpg'):
                    continue
                width = int(1600 * factor)
                height = int(1200 * factor)
                size = int(366 * factor)

                pts1 = np.float32([[width-2, height], [width + size, height-1], [width + size, height + size], [width, height + size]])
                pts2 = np.float32([[width, height], [width + size, height], [width + size, height + size], [width, height + size]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(frame, M, (width, height))
                temp_name = str(base_filename) + '_1_ppct.jpg'
                cv2.imwrite(os.path.join(path + directory, temp_name), dst)


def change_perspective2(path, factor):
    """Change perspective getPerspectiveTransform and using warpPerspective"""
    print('Perspective 2')
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir(path):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(path + directory):
                frame = cv2.imread(os.path.join(path + directory, filename), 1)
                base_filename = os.path.splitext(filename)[0]
                if base_filename.endswith('_ppct'):
                    continue
                width = int(1600 * factor)
                height = int(1200 * factor)
                size = int(366 * factor)

                pts1 = np.float32([[width+2, height], [width + size, height], [width + size-2, height + size+1], [width+1.5, height + size+1]])
                pts2 = np.float32([[width, height], [width + size, height], [width + size, height + size], [width, height + size]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(frame, M, (width, height))
                temp_name = str(base_filename) + '_2_ppct.jpg'
                cv2.imwrite(os.path.join(path + directory, temp_name), dst)


def change_perspective3(path, factor):
    """Change perspective getPerspectiveTransform and using warpPerspective"""
    print('Perspective 3')
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir(path):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(path + directory):
                frame = cv2.imread(os.path.join(path + directory, filename), 1)
                base_filename = os.path.splitext(filename)[0]
                if base_filename.endswith('_ppct'):
                    continue
                width = int(1600 * factor)
                height = int(1200 * factor)
                size = int(366 * factor)

                pts1 = np.float32([[width, height], [width + size, height], [width + size, height + size], [width-2, height + size+2]])
                pts2 = np.float32([[width, height], [width + size, height], [width + size, height + size], [width, height + size]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(frame, M, (width, height))
                temp_name = str(base_filename) + '_3_ppct.jpg'
                cv2.imwrite(os.path.join(path + directory, temp_name), dst)


def change_perspective4(path, factor):
    """Change perspective getPerspectiveTransform and using warpPerspective"""
    print('Perspective 4')
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir(path):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(path + directory):
                frame = cv2.imread(os.path.join(path + directory, filename), 1)
                base_filename = os.path.splitext(filename)[0]
                if base_filename.endswith('_ppct'):
                    continue
                width = int(1600 * factor)
                height = int(1200 * factor)
                size = int(366 * factor)

                pts1 = np.float32([[width, height+2], [width + size, height+4], [width + size+2, height + size], [width, height + size]])
                pts2 = np.float32([[width, height], [width + size, height], [width + size, height + size], [width, height + size]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(frame, M, (width, height))
                temp_name = str(base_filename) + '_4_ppct.jpg'
                cv2.imwrite(os.path.join(path + directory, temp_name), dst)


def change_perspective5(path, factor):
    """Change perspective getPerspectiveTransform and using warpPerspective"""
    print('Perspective 5')
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir(path):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(path + directory):
                frame = cv2.imread(os.path.join(path + directory, filename), 1)
                base_filename = os.path.splitext(filename)[0]
                if base_filename.endswith('_ppct'):
                    continue
                width = int(1600 * factor)
                height = int(1200 * factor)
                size = int(366 * factor)

                pts1 = np.float32([[width+2, height-2], [width + size, height], [width + size+2, height + size], [width, height + size]])
                pts2 = np.float32([[width, height], [width + size, height], [width + size, height + size], [width, height + size]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(frame, M, (width, height))
                temp_name = str(base_filename) + '_5_ppct.jpg'
                cv2.imwrite(os.path.join(path + directory, temp_name), dst)


def change_perspective6(path, factor):
    """Change perspective getPerspectiveTransform and using warpPerspective"""
    print('Perspective 6')
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir(path):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(path + directory):
                frame = cv2.imread(os.path.join(path + directory, filename), 1)
                base_filename = os.path.splitext(filename)[0]
                if base_filename.endswith('_ppct'):
                    continue
                width = int(1600 * factor)
                height = int(1200 * factor)
                size = int(366 * factor)

                pts1 = np.float32([[width, height+1], [width + size, height-2], [width + size, height + size-2], [width+1, height + size]])
                pts2 = np.float32([[width, height], [width + size, height], [width + size, height + size], [width, height + size]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(frame, M, (width, height))
                temp_name = str(base_filename) + '_6_ppct.jpg'
                cv2.imwrite(os.path.join(path + directory, temp_name), dst)


def change_perspective_once(path, factor):
    """Change perspective getPerspectiveTransform and using warpPerspective"""
    print('Perspectives at once')
    for directory in os.listdir(path):
        print(directory)
        for dataset in os.listdir(path + directory):
            print(dataset)
            for filename in os.listdir(path + directory + '/' + dataset):
                print(filename)
                frame = cv2.imread(os.path.join(path + directory + '/' + dataset, filename), 1)
                base_filename = os.path.splitext(filename)[0]
                if base_filename.endswith('_ppct'):
                    continue
                width = int(1600 * factor)
                height = int(1200 * factor)
                size = int(366 * factor)

                # First perspective

                pts1 = np.float32([[width - 2, height], [width + size, height - 1], [width + size, height + size],
                                   [width, height + size]])
                pts2 = np.float32(
                    [[width, height], [width + size, height], [width + size, height + size], [width, height + size]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(frame, M, (width, height))
                temp_name = str(base_filename) + '_1_ppct.jpg'
                cv2.imwrite(os.path.join(path + directory + '/' + dataset, temp_name), dst)

                # Second perspective

                pts1 = np.float32([[width + 2, height], [width + size, height], [width + size - 2, height + size + 1],
                                   [width + 1.5, height + size + 1]])
                pts2 = np.float32(
                    [[width, height], [width + size, height], [width + size, height + size], [width, height + size]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(frame, M, (width, height))
                temp_name = str(base_filename) + '_2_ppct.jpg'
                cv2.imwrite(os.path.join(path + directory + '/' + dataset, temp_name), dst)

                # Third perspective

                pts1 = np.float32([[width, height], [width + size, height], [width + size, height + size],
                                   [width - 2, height + size + 2]])
                pts2 = np.float32(
                    [[width, height], [width + size, height], [width + size, height + size], [width, height + size]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(frame, M, (width, height))
                temp_name = str(base_filename) + '_3_ppct.jpg'
                cv2.imwrite(os.path.join(path + directory + '/' + dataset, temp_name), dst)

                # Fourth perspective

                pts1 = np.float32([[width, height + 2], [width + size, height + 4], [width + size + 2, height + size],
                                   [width, height + size]])
                pts2 = np.float32(
                    [[width, height], [width + size, height], [width + size, height + size], [width, height + size]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(frame, M, (width, height))
                temp_name = str(base_filename) + '_4_ppct.jpg'
                cv2.imwrite(os.path.join(path + directory + '/' + dataset, temp_name), dst)

                # Fifth perspective

                pts1 = np.float32([[width + 2, height - 2], [width + size, height], [width + size + 2, height + size],
                                   [width, height + size]])
                pts2 = np.float32(
                    [[width, height], [width + size, height], [width + size, height + size], [width, height + size]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(frame, M, (width, height))
                temp_name = str(base_filename) + '_5_ppct.jpg'
                cv2.imwrite(os.path.join(path + directory + '/' + dataset, temp_name), dst)

                # Sixth perspective

                pts1 = np.float32([[width, height+1], [width + size, height-2], [width + size, height + size-2],
                                   [width+1, height + size]])
                pts2 = np.float32([[width, height], [width + size, height], [width + size, height + size],
                                   [width, height + size]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(frame, M, (width, height))
                temp_name = str(base_filename) + '_6_ppct.jpg'
                cv2.imwrite(os.path.join(path + directory + '/' + dataset, temp_name), dst)


def remover(path, name):
    """Remove images using it name"""
    print('Removing...')
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir(path):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(path + directory):
                if filename.endswith(name):
                    os.remove(os.path.join(path + directory, filename))


def cropper(path):
    """Crop images according to set values for x & y, width & height"""
    print('Cropping...')
    # dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir(path):
        print(directory)
        for dataset in os.listdir(path + directory):
            print(dataset)
            for filename in os.listdir(path + directory + '/' + dataset):
                print('Crop  ' + filename)
                # base_filename = os.path.splitext(filename)[0]
                frame = cv2.imread(os.path.join(path + directory + '/' + dataset, filename), 1)
                y = 270 # int(500 * factor)  # 80 500*0.1866 200
                x = 360 # int(650 * factor)  # 118 650*0.1866 260
                y_size = 660 # int(300 * factor)  # 64 300*0.1866 120
                x_size = 880
                cropped_image = frame[y:(y + y_size), x:(x + x_size)]
                # base_filename = os.path.splitext(filename)[0]
                # temp_name = str(base_filename) + '.jpg'
                os.remove(os.path.join(path + directory + '/' + dataset, filename))
                cv2.imwrite(os.path.join(path + directory + '/' + dataset, filename), cropped_image)
                # os.rename(os.path.join(path + directory, filename), os.path.join(path + directory, temp_name))


def rotator(path):
    """Rotates images by 90*, 180* and 270*"""
    print('Rotating...')
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir(path):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(path + directory):
                if filename.endswith('cropped.jpg'):
                    frame = cv2.imread(os.path.join(path + directory, filename), 1)
                    frame = cv2.transpose(frame)
                    frame_CV = cv2.flip(frame, 1)
                    frame_CCV = cv2.flip(frame, 2)
                    frame_180 = cv2.flip(frame, -1)
                    base_filename = os.path.splitext(filename)[0]
                    # cv2.imwrite(os.path.join(directory, base_filename + '_CV.jpg'), frame_CV)
                    # cv2.imwrite(os.path.join(directory, base_filename + '_CCV.jpg'), frame_CCV)
                    cv2.imwrite(os.path.join(path + directory, base_filename + '_180.jpg'), frame_180)


def grayer(path):
    """Convert images to grayscale using BGR2GRAY"""
    print('Graying...')
    # dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir(path):
        print(directory)
        for dataset in os.listdir(path + directory):
            print(dataset)
            for filename in os.listdir(path + directory + '/' + dataset):
                print(filename)
                frame = cv2.imread(os.path.join(path + directory + '/' + dataset, filename), 1)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(path + directory + '/' + dataset, filename), gray_frame)


def scaler(path, factor):
    """Resize image to appropriate size"""
    print('Scaling...')
    # dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir(path):
        print(directory)
        for dataset in os.listdir(path + directory):
            print(dataset)
            for filename in os.listdir(path + directory + '/' + dataset):
                print('Sc  ' + filename)
                frame = cv2.imread(os.path.join(path + directory + '/' + dataset, filename), 1)
                scaled_frame = cv2.resize(frame, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(path + directory + '/' + dataset, filename), scaled_frame)


def rotator_angle(path, actual_angle):
    """Rotate images according to angle"""
    print('Rotating...')
    # dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir(path):
        print(directory)
        for dataset in os.listdir(path + directory):
            print(dataset)
            for filename in os.listdir(path + directory + '/' + dataset):
                print('Rot  ' + filename)
                frame = cv2.imread(os.path.join(path + directory + '/' + dataset, filename), 1)
                base_filename = os.path.splitext(filename)[0]
                frame_center = tuple(np.array(frame.shape[1::-1])/2)
                for i in range(0, 360, actual_angle):
                    if i is 0:
                        continue
                    matrix = cv2.getRotationMatrix2D(frame_center, i, 1)
                    result_image = cv2.warpAffine(frame, matrix, frame.shape[1::-1], flags=cv2.INTER_LINEAR)
                    cv2.imwrite(os.path.join(path + directory + '/' + dataset, base_filename + '_' + str(i) + '_rot' + '.jpg'), result_image)


def remove_rotated(path):
    """Remove rotated images"""
    print('Rotating...')
    dir_list = list(map(str, range(1, 7)))
    for directory in os.listdir(path):
        print(directory)
        if directory in dir_list:
            for filename in os.listdir(path + directory):
                if filename.endswith('_rot.jpg'):
                    os.remove(os.path.join(path + directory, filename))


def change_all_perspectives(path, factor):
    change_perspective1(path, factor)
    change_perspective2(path, factor)
    change_perspective3(path, factor)
    change_perspective4(path, factor)
    change_perspective5(path, factor)
    change_perspective6(path, factor)


def all_operations(path, factor):
    print(path)
    # create_empty_dirs(path)
    # name_simpler(path)
    # scaler(path, 0.1866)
    # change_all_perspectives(path, 1)
    # change_perspective_once(path, 1)
    # rotator_angle(path, 30)
    # cropper(path)
    # scaler(path, factor)
    grayer(path)


# all_operations('../kostki/gen_x2/', 0.727272)

# name_simpler('../kostki/oryginalnyRozmiar/redOnRed/')
# name_simpler('../kostki/oryginalnyRozmiar/whiteOnRed_distance/')
# name_simpler('../kostki/oryginalnyRozmiar/whiteOnRed/')
# name_simpler('../kostki/oryginalnyRozmiar/whiteOnBlue/')
# name_simpler('../kostki/oryginalnyRozmiar/woodOnRed/')

# create_dirs()
# create_empty_dirs("original/blue/")
# create_empty_dirs("blue/")
# create_empty_dirs("distance/")
# create_empty_dirs("redOnRed/")
# create_empty_dirs("wood/")
# create_empty_dirs("hardLight/")
# name_simpler("original/blue/")


# name_simpler("hardLight/")
# remover(".")
# copier(".")
# remover('_ppct.jpg')
# cropper(".")
# rotator(".")
# grayer('color/')

# scaler(".")
# scaler("blue/")
# scaler("redOnRed/")
# scaler("wood/")

# remover("blue/", '_ppct.jpg')
# name_changer('blue/', '_scaled.jpg')
# cropper('blue/')
# rotator_angle(".", 5)


# cropper("redOnRed/")
# cropper("wood/")
# cropper("hardLight/")
# remove_rotated("blue/")
# remover('distance/', '_scaled.jpg')
# remove_rotated('distance/')
# remover('wood/', '_scaled.jpg')
# remove_rotated('wood/')
# remover('redOnRed/', '_scaled.jpg')
# remove_rotated('redOnRed/')


# create_empty_dirs("blue/")
# name_simpler("blue/")
# scaler("blue/")
# change_all_perspectives("blue/")
# rotator_angle("blue/", 30)
# cropper("blue/")
# grayer("blue/")


# create_empty_dirs("hardLight/")
# name_simpler("hardLight/")
# scaler("hardLight/")
# change_all_perspectives("hardLight/")
# rotator_angle("hardLight/", 30)
# cropper('hardLight/')
# grayer('hardLight/')


# create_empty_dirs("distance/")
# name_simpler("distance/")
# scaler("distance/")
# change_all_perspectives("distance/")
# rotator_angle("distance/", 30)
# cropper("distance/")
# grayer('distance/')


# create_empty_dirs("wood/")
# name_simpler("wood/")
# scaler("wood/")
# change_all_perspectives("wood/")
# rotator_angle("wood/", 30)
# cropper("wood/")
# grayer('wood/')


# create_empty_dirs("redOnRed/")
# name_simpler("redOnRed/")
# scaler("redOnRed/")
# change_all_perspectives("redOnRed/")
# rotator_angle("redOnRed/", 30)
# cropper("redOnRed/")
# grayer('redOnRed/')


# create_empty_dirs("blackOnRed/")
# name_simpler("blackOnRed/")
# scaler("blackOnRed/")
# change_all_perspectives("blackOnRed/")
# rotator_angle("blackOnRed/", 45)
# cropper("blackOnRed/")
# grayer('blackOnRed/')

# create_empty_dirs("redOnRed_white/")
# name_simpler("redOnRed_white/")
# scaler("redOnRed_white/")
# change_all_perspectives("redOnRed_white/")
# rotator_angle("redOnRed_white/", 45)
# cropper("redOnRed_white/")
# grayer('redOnRed_white/')

# create_empty_dirs("blackOnBlack/")
# create_empty_dirs("whiteOnBlack/")
# create_empty_dirs("greenOnGreen/")
# create_empty_dirs("greenOnWhite/")
# create_empty_dirs("navyOnBlue/")
# create_empty_dirs("navyOnWhite/")
# create_empty_dirs("stainOnWhite/")

# all_operations("blackOnBlack/")
# all_operations("blackOnRed/")
# all_operations("greenOnGreen/")
# all_operations("greenOnWhite/")
# all_operations("hardLight/")
# all_operations("navyOnBlue/")
# all_operations("navyOnWhite/")
# all_operations("redOnRed/")
# all_operations("redOnRed_white/")
# all_operations("stainOnWhite/")
# all_operations("whiteOnBlack/")
# all_operations("whiteOnBlue/")
# all_operations("whiteOnRed/")
# all_operations("whiteOnRed_distance/")
# all_operations("woodOnRed/")

# all_operations("blackOnBlack/")
# all_operations("blackOnRed/")
# all_operations("greenOnGreen/")
# all_operations("greenOnWhite/")
# all_operations("hardLight/")
# all_operations("navyOnBlue/")
# all_operations("navyOnWhite/")
# all_operations("redOnRed/")
# all_operations("redOnRed_white/")
# all_operations("stainOnWhite/")
# all_operations("whiteOnBlack/")
# all_operations("whiteOnBlue/")
# all_operations("whiteOnRed/")
# all_operations("whiteOnRed_distance/")
# all_operations("woodOnRed/")

# all_operations("../zdjeciax4/blackOnRed/", 0.4)
# all_operations("../zdjeciax4/greenOnWhite/", 0.4)
# all_operations("../zdjeciax4/hardLight/", 0.4)
# all_operations("../zdjeciax4/navyOnBlue/", 0.4)
# all_operations("../zdjeciax4/navyOnWhite/", 0.4)
# all_operations("../zdjeciax4/whiteOnBlack/", 0.4)
# all_operations("../zdjeciax4/whiteOnBlue/", 0.4)
# all_operations("../zdjeciax4/whiteOnRed/", 0.4)
# all_operations("../zdjeciax4/whiteOnRed_distance/", 0.4)
# all_operations("../zdjeciax4/woodOnRed/", 0.4)

# windowname = 'Image Processing'
# cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
# image = 'original/2/1.jpg'
#
# originalFrame = cv2.imread(image, 1)
# if originalFrame is None:
#     raise Exception('cannot read')
#
# originalFrame = cv2.resize(originalFrame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
#
# # Change originalFrame to grayscale
# processedFrame = cv2.cvtColor(originalFrame, cv2.COLOR_BGR2GRAY)


# morphological transformations
# kernel = np.ones((2, 2), np.uint8)

# erosion
# erosion = cv2.erode(processedFrame, kernel, iterations=1)
# dilation
# dilation = cv2.dilate(processedFrame, kernel, iterations=1)
# opening
# opening = cv2.morphologyEx(processedFrame, cv2.MORPH_OPEN, kernel)

# Display the results
# y = 500
# x = 650
# size = 360
#
# y /= 4
# x /= 4
# size /= 4
# x, y, size = int(x), int(y), int(size)
#
#
# pts1 = np.float32([[y, x], [y+size, x], [y+size, x+size], [y, x+size]])
# pts11 = np.float32([[y, x], [y+size+10, x], [y+size, x+size], [y, x+size]])
# pts2 = np.float32([[y-20, x-20], [y+size+20, x+20], [y+size+20, x+size-20], [y-20, x+size+20]])
#
#
# M = cv2.getPerspectiveTransform(pts1, pts11)
# dst = cv2.warpPerspective(processedFrame, M, (400, 300))
# cropped_image = dst[y:(y + size), x:(x + size)]
# # cv2.imshow(windowname, cropped_image)
# cv2.imshow(windowname, dst)
# # cv2.waitKey(0)
# # Release capturing video
# cv2.destroyAllWindows()
