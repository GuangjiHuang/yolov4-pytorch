import cv2 as cv
import os
import time

os.chdir(os.path.dirname(__file__))

def save_img_from_video(video_path=None, save_img_dir=None):
    # deal with the dir and path
    if video_path is None:
        video_path = r"/home/ysq/data/video/up-3.49min.mp4"
    if save_img_dir is None:
        save_img_dir = r"./img"
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)
    #
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("the video can not open!")
    # get the image
    while True:
        ret, frame = cap.read()
        if not ret:
            print("video read out!")
            break
        # try to show the frame 
        cv.imshow("show", frame)
        key_val = cv.waitKey() & 0xff
        if key_val == ord('q'):
            break
        elif key_val == ord('s'):
            file_name = time.strftime("%m-%d-%H-%M-%S", time.localtime()) + ".jpg"
            file_path = os.path.join(save_img_dir, file_name)
            print("save the file name: ",  file_path)
            cv.imwrite(file_path, frame)
def on_mouse_action(event, x, y, flags, parameter):
    global img, pt
    if event == cv.EVENT_LBUTTONUP:
        # show the point
        cv.circle(img, (x, y), 3, (0, 0, 255), thickness=2)
        if len(pt) == 2:
            pt.clear()
        pt.append((x, y))
        cv.imshow("show", img)
        cv.waitKey()
    elif event == cv.EVENT_MBUTTONUP:
        # draw the rectangle
        if len(pt) == 2:
            pt1, pt2 = pt
            cv.rectangle(img, pt1, pt2, (255, 255, 0), 3)
            cv.imshow("show", img)
            key_val = cv.waitKey() & 0xff
            # determine to save or not
            if key_val == ord('s'):
                print("save!")
                save_path = f"./img/{pt1}-{pt2}.jpg"
                x_l, x_r = min(pt1[0], pt2[0]), max(pt1[0], pt2[0])
                y_l, y_r = min(pt1[1], pt2[1]), max(pt1[1], pt2[1])
                save_img = img[y_l: y_r+1, x_l: x_r+1]
                cv.imwrite(save_path, save_img)
                #cv.imshow("cut part", save_img)
                #cv.waitKey()
        else:
            print("Less than two point to draw the recentagle")
            print(pt)

# use the on mouse event to select the roi area
img_path = r"./img/background.jpg"
img = cv.imread(img_path)
o_h, o_w = img.shape[:2]
# resize the image
#img = cv.resize(img, (int(0.5*o_w), int(0.5*o_h)))
pt = list()
window_name = "show"
cv.namedWindow(window_name, cv.WINDOW_KEEPRATIO)
cv.setMouseCallback(window_name, on_mouse_action)
# show the image and then select the ROI
cv.imshow(window_name, img)
cv.waitKey()