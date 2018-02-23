import cv2
import datetime

def capture():
    cap = cv2.VideoCapture(1)
    window_name = 'capture'
    cv2.namedWindow(window_name)

    folder = 'captured/'
    while True:
        ret, frame = cap.read()

        second_frame = frame.copy()
        y = 200
        x = 260
        size = 120

        cv2.rectangle(second_frame, (x, y), (x+size, y+size), (255, 0, 255), 2)

        cv2.imshow(window_name, second_frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            break
        elif k % 256 == 32:
            current_time_date = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
            img_name = current_time_date + '.jpg'
            cv2.imwrite(folder + img_name, frame)
            print('Captured  {}'.format(img_name))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture()