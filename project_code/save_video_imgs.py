import cv2
import numpy as np
import os

video_path = r"/home/ysq/data/video/up-9.17min.mp4"

save_path = r"../img_s/clip_images"
if not os.path.exists(save_path):
    os.mkdir(save_path)

# frame id to be saved
#frames_id = [8917, 9100, 9118, 9126, 9138, 9152, 9464]
frames_id = [1065]

# read the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("can not open the video! Please check the path!")
    exit(0)

total_id = cap.get(cv2.CAP_PROP_FRAME_COUNT)
for frame_id in frames_id:
    if frame_id>=total_id:
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, img = cap.read()
    # write the image
    img_save_path = f"{save_path}/{frame_id}.jpg"
    print(f"save the {img_save_path}")
    cv2.imwrite(img_save_path, img)

cap.release()
