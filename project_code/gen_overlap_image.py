import cv2 as cv, cv2
import numpy as np
import os
import random
from glob import glob
import time
from paste_img import *

def getInsertWH(area_rate, box):
    h, w = box.shape[:2]
    area = h * w
    i_area = area * area_rate
    i_h = random.randint(10, int(h))
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
    b_box = 255 * np.ones((hbb, wbb, 3), np.uint8)
    # get the background
    #img_path = r"./img/(1050, 807)-(1454, 1058).jpg"
    #img = cv2.imread(img_path)
    #b_box = img[10:hbb+10, 10:wbb+10]
    # 
    hide_direction_is_right = random.randint(0, 1)
    if hide_direction_is_right:
        # get the center point
        xa, ya = wbb - wa//2, ha//2
        xb, yb = wb//2, hbb-hb//2
        # renew the b_box
        b_box[:ha, wb-wi:] = box_a
        b_box[ha-hi:, :wb] = box_b
    else:
        xa, ya = wa//2, ha//2
        xb, yb = wbb-wb//2, hbb-hb//2
        b_box[:ha, :wa] = box_a
        b_box[hbb-hb:, wbb-wb:] = box_b

    # if draw the rectangle
    if is_with_rect:
        of = 8
        a_pt1, a_pt2 = (xa-wa//2+of, ya-ha//2+of), (xa+wa//2-of, ya+ha//2-of)
        b_pt1, b_pt2 = (xb-wb//2+of, yb-hb//2+of), (xb+wb//2-of, yb+hb//2-of)
        thickness = 2
        cv2.rectangle(b_box, a_pt1, a_pt2, (0, 0, 255), thickness)
        cv2.rectangle(b_box, b_pt1, b_pt2, (0, 0, 255), thickness)

    return b_box, [(xa, ya, wa, ha), (xb, yb, wb, hb)]
    #return img


def getHeads(root_dir):
    imgs = glob(os.path.join(root_dir, "*.jpg"))
    anns = glob(os.path.join(root_dir, "*.txt"))
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
        for ann in anns:
            cx, cy = ann[0], ann[1]
            x1, y1 = max(0, int(cx-ann[2]/2)), max(0, int(cy-ann[3]/2))
            x2, y2 = min(w, int(cx+ann[2]/2)), min(h, int(cy+ann[3]/2))
            all_bboxes.append(img[y1:y2, x1:x2, :])
            # debug
            #cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255))
        #
    #
    return all_bboxes


if __name__ == "__main__":
    # get the head boxes
    os.chdir(os.path.dirname(__file__))
    root = r"/home/ysq/data/image/image_mark/"
    root_dir_ls = glob(root + "12_NVR*")
    save_dir = r"../img_s/inter_sec_another/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    #
    all_bboxes = list()
    for root_dir in root_dir_ls:
        all_bboxes += getHeads(root_dir)
        break

    print(type(all_bboxes), "len: ", len(all_bboxes))

    # the area_rate range and the number generate
    image_generator_num = 100
    board_img_path = r"./img/(998, 492)-(1548, 1066).jpg"
    beneath_img_path = r"./img/background.jpg"
    beneath_img = cv.imread(beneath_img_path)
    rec_board = parseFileName(board_img_path)
    board_img = cv.imread(board_img_path)
    wh_board = (board_img.shape[1], board_img.shape[0])
    for i in range(image_generator_num):
        # get the overlap bbox
        num_overlap_bbox = random.randint(5, 15)
        overlap_bbox_ls = list()
        overlap_anos_ls = list()
        overlap_wh_ls = list()
        for j in range(num_overlap_bbox):
            box1, box2 = random.sample(all_bboxes, 2)
            b1_h, b1_w = box1.shape[:2]
            b2_h, b2_w = box2.shape[:2]
            #
            resize_flag = True
            if resize_flag:
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
            else:
                box_a, box_b = box1, box2
            #
            #
            area_rate = random.randint(4, 6) / 10
            b_box, anos = genBox(box_b, box_a, area_rate, False)
            overlap_bbox_ls.append(b_box)
            overlap_anos_ls.append(anos)
            overlap_wh_ls.append([b_box.shape[1], b_box.shape[0]]) # the wh
        # paste the overlap bbox to the board
        img_grid = np.copy(board_img)
        rec_ls = list()
        img_overlap_ls = list()
        anos_overlap_ls = list()
        for k in range(len(overlap_wh_ls)):
            wh_paper = overlap_wh_ls[k]
            if not isValidWH((10, 10), wh_paper, wh_board):
                continue
            # get the other
            is_paste = isPaste(wh_paper, wh_board, rec_ls)
            img_overlap_ls.append(overlap_bbox_ls[k])
            anos_overlap_ls.append(overlap_anos_ls[k])
            if not is_paste:
                img_overlap_ls.pop()
                anos_overlap_ls.pop()
                pasteImage(beneath_img, board_img, rec_ls, img_overlap_ls, anos_overlap_ls, rec_board)
                img_overlap_ls.clear()
                anos_overlap_ls.clear()
                rec_ls.clear()
        #
        if len(rec_ls) > 0:
            pasteImage(beneath_img, board_img, rec_ls, img_overlap_ls, anos_overlap_ls, rec_board)
