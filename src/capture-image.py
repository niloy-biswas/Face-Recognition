import cv2
import os

BASE_PATH = os.path.dirname(__file__) + '/images/'
name = input("Please input your name:")

if os.path.exists(BASE_PATH + name):
    print("User is already exists try another")


else:
    os.mkdir(BASE_PATH + name)

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    elif k%256 == 32:
        # SPACE pressed

        path = os.path.join(BASE_PATH + name + '/')
        img_name = "picture_{}.png".format(img_counter)
        cv2.imwrite(path + img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1


cam.release()
cv2.destroyAllWindows()