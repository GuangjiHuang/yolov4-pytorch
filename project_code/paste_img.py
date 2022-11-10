# give you:
# img: w,h
# a series of the rectangles, and then you can put them into the big rectangle 

import random
import numpy as np
import cv2 as cv


def wh_is_valid(start_point, wh_paper, wh_board):
    return wh_paper[0]+start_point[0]<=wh_board[0] and wh_board[1]+start_point[1]<=wh_l[1]

def pos_rectangle(wh_paper, wh_board, used_lines):
    offset = 10
    # return the start_point
    # renew the used_lines
    if len(used_lines) == 0:
        start_point = (10, 10)
        return start_point if wh_is_valid(start_point, wh_paper, wh_board) else None
    # each line use the one list to store the point [(x1, y1, x2, y2)]
    else:
        for i, used_line in enumerate(used_lines):
            # get the right point 
            x1, y1, x2, y2 = used_line[-1] # (x1, y1, x2, y2)
            start_point_x = offset + x2
            start_point_y_min = y1
            y2_ls =  [used_lines[i] for i in range(0, i)]
            start_point_y_max = max(y2_ls) if len(y2_ls)>0 else offset
            # try the start_point_y in the range [start_point_y_min, start_point_y_max]
            for start_point_y in range(start_point_y_min, start_point_y_max, 5):
                start_point = (start_point_x, start_point_y)
                is_valid_flag = wh_is_valid(start_point, wh_paper, wh_board)
                if is_valid_flag:
                    used_line.append((*start_point, start_point[0]+wh_paper[0], start_point+wh_paper[1]))
                    return start_point
        # new the another line
        start_point_x = offset
        start_point_y = start_point_y_max
        start_point = (start_point_x, start_point_y)
        return start_point if wh_is_valid(start_point, wh_paper, wh_board) else None


random.seed(0)
# generate the big image
img_h, img_w = 200, 200
img = 122 * np.ones((img_h, img_w), np.uint8)
cv.imshow("show", img)
cv.waitKey(0)

# generate a series small image
paste_imgs = list()
paste_img_num = 100
for i in range(paste_img_num):
    tmp = list()
    rand_w = random.randint(20, img_w)
    rand_h = random.randint(20, img_h)
    tmp.append((rand_h, rand_w))
    color = list()
    for i in range(3):
        color.append(random.randint(0, 255))
    color = tuple(color)
    tmp.append(color)
    paste_imgs.append(tmp)

for wh in paste_imgs:
    #print(wh)
    pass
# select the starting point
img_grid = np.copy(img)
used_lines = list()
wh_board = (img_w, img_h)
for whc in paste_imgs:
    wh_paper = whc[0]
    # get the others
    start_point = pos_rectangle(wh_board, wh_paper, used_lines)
    if start_point is None:
        used_lines.clear()
        start_point = pos_rectangle(wh_board, wh_paper, used_lines)
        print(start_point)
    else:
        print(start_point)