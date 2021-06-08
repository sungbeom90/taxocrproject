from tests.helpers import load_data
from tests.helpers import img_prepro
import os
import cv2
import numpy as np


# 훈련 데이터 불러오기
file_path = "./data/image/origin_image/"  # 원본파일 경로
file_list_jpg = load_data.load_images(file_path)  # 원본파일 이름 목록 가져오기
print(file_list_jpg)


for jpg_file in file_list_jpg:
    jpg_file_name = file_path + jpg_file  # 파일 경로
    print("preprossing...")
    img = img_prepro.detection_preprocess(jpg_file_name)
    test_data, _ = img_prepro.load_single_img_resize(img, 1600, 1600)
    # # 파일 저장
    # print("file storage 내부 : ", f)
    # f.save("./app/static/image/" + secure_filename(f.filename))
    # upfile_address = "./app/static/image/" + secure_filename(f.filename)


# char_list = Define_Class.definition_class()  # 텍스트 클래스 종합 파일 로드
# model = "./data/trained_weights/1600_pdfdata_origin_d05_decay1000_1600to1600_20210213-2203_last"  # 디텍션 모델 가중치 로드
# model_recog = "./data/trained_weights/tax_save_model_0309.hdf5"  # 리코그니션 모델 가중치 로드
# upfilename = upload_name_list.pop()
# jpg_file_name = upload_file_list.pop()  # 입력 이미지 경로 로드

# print("Detecting...")  # 이미지 디텍팅 실행
# or_image, boxed_image, word_box = pred_detection(jpg_file_name, model, size=1600)

# print(len(word_box))  # 단어단위 디텍션 완성 좌표값 저장 갯수보기 (xmin, ymin, xmax, ymax) 구조
# print(word_box)
