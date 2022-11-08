import os
import cv2

video_path = r"/home/ysq/data/video/up-9.17min.mp4"
save_path = r"../save_videos/s_up-9.17min.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#size = (int(0.5*w), int(0.5*h))
size = int(0.5 * w), int(0.5 * h)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(save_path, fourcc, fps, size)

if not cap.isOpened():
    print("can not read the video!")
    exit(1)
frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("read ending!")
        break
    frame = cv2.resize(frame, size)
    t = int((frame_id+1) / fps)
    t_m =  t // 60
    t_s = t % 60
    cv2.putText(frame, f"frame_id: {frame_id} time: {t_m}:{t_s}", (0, int(0.5*size[1])), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255))
    out.write(frame)
    #print(f"Has been written {frame_id}")
    #cv2.imshow("show", frame)
    #key_val = cv2.waitKey(0) & 0xff
    #if key_val == ord('q'):
    #    break

    frame_id += 1

cap.release()
out.release()