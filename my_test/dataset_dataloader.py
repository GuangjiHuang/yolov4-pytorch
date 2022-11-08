import cv2
import numpy as np
import os
import math
import glob
from PIL import Image
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from nets import yolo
from nets import yolo_training

class MyDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        self.images_no_ext = self._getImages() # the list, images path without the extension


    def __getitem__(self, index):
        # get the image_path and the target_path
        file = self.images_no_ext[index]
        image_path = f"{self.root_dir}/{file}.jpg"
        target_path = f"{self.root_dir}/{file}.txt"

        # read image
        image = cv2.imread(image_path)

        # read the bounding boxes
        with open(target_path, "r") as f:
            target_lines = f.readlines()
        #
        target = [list(map(float, line.strip().split())) for line in target_lines]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resize the image
        if self.transforms is not None:
            image = self.transforms(image)

        return image, target, file


    def __len__(self):
        return len(self.images_no_ext)

    def _getImages(self):
        images_path = glob.glob(os.path.join(self.root_dir, "*.jpg"))
        targets_path = glob.glob(os.path.join(self.root_dir, "*.txt"))
        # get the file name
        images_path_ = [os.path.basename(i).split(".")[0] for i in images_path]
        targets_path_ = [os.path.basename(i).split(".")[0] for i in targets_path]
        # check the images_path
        files_path = [i for i in images_path_ if i in targets_path_]
        return files_path

def readAnchors(anchors_path):
    with open(anchors_path, "r") as f:
        line = f.readline()
    anchors = list(map(float, line.split(",")))
    anchors = np.array(anchors).reshape((-1, 2))
    return anchors

def imgTransform(y_wh, img, boxes):
    h, w = img.shape[:2]
    # scale
    scale = y_wh / max(w, h)
    s_w = min(y_wh, math.ceil(w*scale))
    s_h = min(y_wh, math.ceil(h*scale))
    # resize
    s_img = cv2.resize(img, (s_w, s_h))
    # dx, and dy
    dx, dy = int((y_wh-s_w)/2), int((y_wh-s_h)/2)
    # add the box
    y_img = np.zeros((y_wh, y_wh, 3)).astype(img.dtype)
    y_img[dy:s_h+dy, dx:s_w+dx, :] = s_img
    # deal the boxes
    boxes = boxes * scale
    boxes[:, 1] = boxes[:, 1] + dx
    boxes[:, 2] = boxes[:, 2] + dy
    boxes = np.ceil(boxes)
    #boxes = np.where(boxes>y_wh, y_wh, boxes)
    return y_img, boxes

def calIou(boxes, anchors):
    bs_anchors = list()
    # move the x, y to the 0
    for box in boxes[:, 3:]:
        in_box = np.where(box<=anchors, box, anchors)
        inser_area = in_box[:, 0]*in_box[:, 1]
        union_area = box[0]*box[1] + anchors[:, 0]*anchors[:, 1] - inser_area
        union_area = np.where(union_area<=0, 1e-6, union_area)
        box_iou = inser_area / union_area
        # add the bs_anchor
        bs_anchor = np.argmax(box_iou, 0)
        bs_anchors.append(bs_anchor)

    return np.array(bs_anchors, dtype=np.uint8)

def getGridCenter(bboxes, anchor_scales, i_wh):
    anchor_scales = np.expand_dims(anchor_scales, 1)
    cx, cy = bboxes[:, 1], bboxes[:, 2]
    ij = bboxes[:,1:3] / i_wh * anchor_scales
    ij = np.floor(ij) + 0.5 # the center
    return ij / anchor_scales * i_wh

def visImage(dataset, y_wh, anchors):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # check if the path exists
    save_dir = r"../save_videos/"
    if not os.path.exists(save_dir):
        print(f"{save_dir} not exist! Making dir!")
        os.mkdir(save_dir)
    save_path = save_dir + "save.mp4"
    fps = 2
    size = (416*2, 416*2)
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, size)
    frame_id = 0
    for i in range(len(dataset)):
        frame_id += 1
        if frame_id > 30:
            break
        # draw the bboxes in the picture
        img, bboxes, file_name = dataset[i]
        if len(bboxes) == 0:
            continue
        h, w = img.shape[:2]
        bboxes = np.array(bboxes)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * w
        bboxes[:, [2, 4]] = bboxes[:, [2, 4]] * h
        bboxes = np.ceil(bboxes)
        img, bboxes = imgTransform(y_wh, img, bboxes)
        # get the anchors id and the scale
        bs_anchors = calIou(bboxes, anchors)
        anchor_scales = 52 / np.power(2, (bs_anchors // 3))
        grid_c = getGridCenter(bboxes, anchor_scales, y_wh)
        bb_anchors = np.concatenate((grid_c, anchors[bs_anchors]), 1)

        for i in range(len(bboxes)):
            # the bbox
            cx, cy = bboxes[i][1], bboxes[i][2]
            bw, bh = bboxes[i][3], bboxes[i][4]
            pt1 = (max(int(cx-bw/2), 0), max(int(cy-bh/2), 0))
            pt2 = (min(int(cx+bw/2), y_wh), min(int(cy+bh/2), y_wh))
            # the bb_anchor
            #acx, acy = bb_anchors[i][0], bb_anchors[i][1]
            abw, abh = bb_anchors[i][2], bb_anchors[i][3]
            acx, acy = cx, cy
            apt1 = (max(int(acx-abw/2), 0), max(int(acy-abh/2), 0))
            apt2 = (min(int(acx+abw/2), y_wh), min(int(acy+abh/2), y_wh))
            # draw the rectangle
            cv2.rectangle(img, pt1, pt2, (0, 0, 255))
            cv2.rectangle(img, apt1, apt2, (0, 255, 255))

        # show the image with bboxes
        # put the text
        img = cv2.resize(img, size)
        cv2.putText(img, "red: ground truth", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
        cv2.putText(img, "yellow: best anchor", (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255))
        # write the frame to the video
        video_writer.write(img)
        cv2.imshow("show", img)
        key_val = cv2.waitKey() & 0xff
        if key_val == ord('q'):
            video_writer.release()
            print("exit!")
            break

        # save the image
        #file_path = f"../save_images/{file_name}.jpg"
        ##cv2.imwrite(file_path, img)
        #print(f"has beend saved {file_path}")
    video_writer.release()


if __name__ == "__main__" :
    dir = r"../VOCdevkit/head/head_data/obj/"
    anchors_path = r"../model_data/yolo_anchors.txt"
    anchors = readAnchors(anchors_path)
    person_dataset = MyDataset(dir)
    visImage(person_dataset, 416, anchors)
