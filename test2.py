import pickle
import cv2
import numpy as np
from tests.detection import decoding
from tests.detection import iou

with open("./data/pickle/heatmap_0", "rb") as file:  # james.p 파일을 바이너리 읽기 모드(rb)로 열기
    data = pickle.load(file)

a = decoding.fun_decoding(data)
print(a)

list = a

precision, recall = iou.TP_check(list,list)
print("precision : {}, recall : {}".format(precision, recall))

