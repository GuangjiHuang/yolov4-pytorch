import cv2
import numpy as np
import os
import random
import glob
import time

def getInsertWH(area_rate, box):
    h, w = box.shape[:2]
    area = h * w
    i_area = area * area_rate
    i_h = random.randint(1, int(h))
    i_w = min(int(i_area // i_h), w-1)
    return i_h, i_w

def genBox(box_b, box_a, area_rate, is_with_rect=False):
    # clone the box_a
    box_a = box_a.copy()
    hb, wb = box_b.shape[:2]
    ha, wa = box_a.shape[:2]

    # get the insert width and height
    hi, wi = getInsertWH(area_rate, box_a)
    hi, wi = min(hi, hb), min(wi, wb)

    # the b_box
    hbb = hb + ha - hi
    wbb = wb + wa - wi
    b_box = 122 * np.ones((hbb, wbb, 3), np.uint8)
    # get the center point
    xa, ya = wbb - wa//2, ha//2
    xb, yb = wb//2, hbb-hb//2

    # mask the box_a
    box_a[ha-hi:, :wi, :] = 0

    # renew the b_box
    b_box[:ha, wb-wi:] = box_a
    b_box[ha-hi:, :wb] = box_b

    # if draw the rectangle
    if is_with_rect:
        of = 8
        a_pt1, a_pt2 = (xa-wa//2+of, ya-ha//2+of), (xa+wa//2-of, ya+ha//2-of)
        b_pt1, b_pt2 = (xb-wb//2+of, yb-hb//2+of), (xb+wb//2-of, yb+hb//2-of)
        thickness = 2
        cv2.rectangle(b_box, a_pt1, a_pt2, (0, 0, 255), thickness)
        cv2.rectangle(b_box, b_pt1, b_pt2, (0, 0, 255), thickness)

    return b_box


def getHeads(root_dir):
    imgs = glob.glob(os.path.join(root_dir, "*.jpg"))
    anns = glob.glob(os.path.join(root_dir, "*.txt"))
    f_imgs = [os.path.basename(img)[:-4] for img in imgs]
    f_ans = [os.path.basename(ann)[:-4] for ann in anns]
    # check if f_img in the f_ans
    files = [i for i in f_imgs if i in f_ans] # just the files name
    # read the img and the anns
    all_anns = list()
    all_bboxes = list()
    for file in files:
        ann_path = os.path.join(root_dir, file+".txt")
        img_path = os.path.join(root_dir, file+".jpg")
        # the ann
        with open(ann_path, "r") as f:
            anns = f.readlines()
        anns = [list(map(float, line.split())) for line in anns]
        anns = np.array(anns)[:, 1:] # just get the x,y,w,h
        if len(anns) == 0:
            continue
        # read the image and then get the box
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        anns[:, [0,2]] = anns[:, [0,2]] * w
        anns[:, [1,3]] = anns[:, [1,3]] * h
        # to the pt1, pt2 form
        #anns[:, 0:2] = anns[:, 0:2] - anns[:, 2:4] / 2
        #anns[:, 2:4] = anns[:, 0:2] + anns[:, 2:4]
        #anns = np.ceil(anns)
        #anns[:, 0:2] = np.where(anns[:, 0:2]<0, 0, anns[:, 0:2])
        #anns[:, 2] = np.where(anns[:, 2]>w, w, anns[:, 2])
        #anns[:, 3] = np.where(anns[:, 3]>h, h, anns[:, 3])
        #anns = anns.astype(np.int32)
        all_anns.append(anns)
        # boxes_part
        boxes = list()
        for ann in anns:
            cx, cy = ann[0], ann[1]
            x1, y1 = max(0, int(cx-ann[2]/2)), max(0, int(cy-ann[3]/2))
            x2, y2 = min(w, int(cx+ann[2]/2)), min(h, int(cy+ann[3]/2))
            boxes.append(img[y1:y2, x1:x2, :])
            # debug
            #cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255))
        #
        all_bboxes.append(boxes)
    #
    return all_bboxes, all_anns



#for boxes in all_bboxes:
#    for box in boxes:
#        cv2.imshow("show", box)
#        key_val = cv2.waitKey(0) & 0xff
#        if key_val == ord("q"):
#            exit(0)

if __name__ == "__main__":
    # get the head boxes
    root_dir = r"/home/ysq/data/image/image_mark/12_NVR_IPC_20200213072959_20200213075124_2138410_5.17min_clip"
    save_dir = r"../img_s/inter_sec/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    #
    all_bboxes, all_anns = getHeads(root_dir)

    # get the box_b and the box_a
    box_b = np.zeros((180, 150, 3), np.uint8)
    box_b[..., 2] = 255
    box_a = np.zeros((150, 120, 3), np.uint8)
    box_a[..., 0] = 255
    o_box_a = box_a
    o_box_b = box_b

    # the area_rate range and the number generate
    while True:
        # get the box1
        box1 = random.choice(all_bboxes)
        box1 =random.choice(box1)
        b1_h, b1_w = box1.shape[:2]
        # get the box2
        box2 =random.choice(all_bboxes)
        box2 = random.choice(box2)
        b2_h, b2_w = box2.shape[:2]
        #
        if b1_h*b1_w > b2_h*b2_w:
            scale = b1_h / b2_h
            b2_w = int(b2_w * scale)
            box2 = cv2.resize(box2, (b2_w, b1_h))
            box_a, box_b = box2, box1
        else:
            scale = b2_h / b1_h
            b1_w = int(b1_w * scale)
            box1 = cv2.resize(box1, (b2_h, b1_w))
            box_a, box_b = box1, box2
        #
        just_red_and_blue = True
        if just_red_and_blue:
            box_a = o_box_a
            box_b = o_box_b
        #
        area_rate = random.randint(4, 6) / 10
        b_box = genBox(box_b, box_a, area_rate, False)

        # show the b_box
        cv2.imshow("show", b_box)
        key_val = cv2.waitKey() & 0xff
        if key_val == ord("q"):
            break
        elif key_val == ord("s"):
            print("save the image!")
            filename = time.strftime("%y-%m-%d-%H-%M-%S", time.localtime())
            cv2.imwrite(filename+".jpg", b_box)

        # past the b_box to the image
