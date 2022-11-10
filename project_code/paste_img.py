# give you:
# img: w,h
# a series of the rectangles, and then you can put them into the big rectangle 

import random
import time
import numpy as np
import cv2 as cv
from glob import glob
import os
import re

def showPaste(img, rec_ls):
    img_paste = np.copy(img)
    for rec in rec_ls:
        pt1 = (rec[0], rec[1])
        pt2 = (rec[2], rec[3])
        cv.rectangle(img_paste, pt1, pt2, (0, 0, 255),  thickness=cv.FILLED)
    # then show the image
    cv.namedWindow("paste", cv.WINDOW_KEEPRATIO)
    cv.imshow("paste", img_paste)
    key_val = cv.waitKey() & 0xff
    if key_val == ord('q'):
        pass
def parseFileName(file_name):
    pattern = re.compile(r"\d+")
    ret_ls = re.findall(pattern, file_name)
    rec = list(map(int, ret_ls))
    return rec if len(rec) == 4 else []
def pasteImage(img_beneath, img_board, rec_ls, img_overlap, anos_overlap, rec_board):
    img_bo = np.copy(img_board)
    img_be = np.copy(img_beneath)
    ox1, oy1, ox2, oy2 = rec_board
    for i in range(len(rec_ls)):
        rec = rec_ls[i]
        x1, y1, x2, y2 = rec
        # renew the anos
        ba_cx, ba_cy, ba_w, ba_h = anos_overlap[i][0]
        bb_cx, bb_cy, bb_w, bb_h = anos_overlap[i][1]
        ba_cx, ba_cy = ba_cx+x1+ox1, ba_cy+y1+oy1
        bb_cx, bb_cy = bb_cx+x1+ox1, bb_cy+y1+oy1
        img_bo[y1:y2, x1:x2, :] = img_overlap[i]
    img_be[oy1:oy2+1, ox1:ox2+1] = img_bo
    # show the image
    is_show = False
    if is_show:
        cv.namedWindow("paste", cv.WINDOW_KEEPRATIO)
        cv.imshow("paste", img_be)
        cv.waitKey()
    # save the image
    file_dir = f"./img/overlap_img/"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name_prefix = time.strftime("%d-%H-%M-%S", time.localtime())
    is_save = True
    if is_save:
        cv.imwrite(file_name_prefix+".jpg", img_be)
    content = list()
    for anos in anos_overlap:
        content.append(" ".join(list(anos)))
    content = "\n".join(content)
    with open(file_name_prefix+".txt", "w") as f:
        f.write(content)

def isValidWH(start_point, wh_paper, wh_board):
    return wh_paper[0]+start_point[0]<=wh_board[0] and wh_paper[1]+start_point[1]<=wh_board[1]

def isRecCross(rec1, rec2):
    x11, y11, x12, y12 = rec1
    x21, y21, x22, y22 = rec2
    #
    x = abs(x11-x12) + abs(x21-x22)
    y = abs(y11-y12) + abs(y21-y22)
    #
    cx = abs(x11+x12-x21-x22)
    cy = abs(y11+y12-y21-y22)
    return cx<=x and cy<=y

def isPaste(wh_paper, wh_board, rec_ls):
    # seting the constant
    edge_gap = random.randint(10, 20)
    grid_gap = random.randint(10, 30)
    # 
    w_p, h_p = wh_paper
    w_b, h_b = wh_board
    is_paste = False
    for y in range(edge_gap, h_b, grid_gap):
        for x in range(edge_gap, w_b, grid_gap):
            # check is valid wh
            if not isValidWH((x, y), wh_paper, wh_board):
                continue
            #
            i = 0
            possible_rec = [x, y, x+w_p, y+h_p]
            # check if possible_rec cross with the other rec
            for rec in rec_ls:
                if isRecCross(possible_rec, rec):
                    break
                i += 1
            # if not cross, set is_paste flag, then add possible_rec to the rec_ls
            if i == len(rec_ls):
                is_paste = True
                # add the rec to the rec_ls
                rec_ls.append(possible_rec)
                return True
    # return the start_point
    return is_paste

if __name__ == "__main__":
    random.seed(0)
    # generate the big image
    img_h, img_w = 200, 200
    img = 0 * np.ones((img_h, img_w, 3), np.uint8)
    # read the background
    img_path = "./img/(1004, 435)-(1799, 1055).jpg"
    img = cv.imread(img_path)
    img_h, img_w = img.shape[:2]
    #cv.imshow("show", img)
    #cv.waitKey(10)

    # generate a series small image
    paste_imgs = list()
    paste_img_num = 100
    for i in range(paste_img_num):
        tmp = list()
        rand_w = random.randint(5, int(0.4*img_w))
        rand_h = random.randint(5, int(0.5*img_h))
        tmp.append((rand_h, rand_w))
        color = list()
        for i in range(3):
            color.append(random.randint(0, 255))
        color = tuple(color)
        tmp.append(color)
        paste_imgs.append(tmp)
    # use the paste_img
    paste_imgs.clear()
    paste_img_dir = r"../img_s/inter_sec/"
    file_path_ls = glob(paste_img_dir + "*.jpg")
    for file_path in file_path_ls:
        paste_img = cv.imread(file_path)
        paste_imgs.append([*paste_img.shape[:2][::-1], file_path])
    #
    for wh in paste_imgs:
        print(wh)
        pass
    # select the starting point
    img_grid = np.copy(img)
    rec_ls = list()
    wh_board = (img_w, img_h)
    for whp in paste_imgs:
        wh_paper = whp[:-1]
        # check the shape size first
        if not isValidWH((10, 10), wh_paper, wh_board):
            print("Error: image shape too big! Strip")
            continue
        # get the others
        is_paste = isPaste(whp, wh_board, rec_ls)
        if not is_paste:
            # can not put the whc into the wh_board
            # show the result, new the board
            pasteImage(img, rec_ls)
            rec_ls.clear()
            # paste again
            isPaste(whp, wh_board, rec_ls)
    # show the last time dealing
    pasteImage(img, rec_ls)