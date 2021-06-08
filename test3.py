from tests.helpers import load_data
from tests.helpers import img_prepro
import os
import cv2
import numpy as np
import pickle


# 훈련 데이터 불러오기
file_path = "./data/image/original_image/"  # 원본파일 경로
file_list_jpg = load_data.load_images(file_path, ",jpg")  # 원본파일 이름 목록 가져오기
print(file_list_jpg)


for jpg_file in file_list_jpg:
    jpg_file_name = file_path + jpg_file  # 파일 경로
    print("preprossing...")
    bw_img = img_prepro.detection_preprocess(jpg_file_name)
    img_prepro.imshow("bw_img", bw_img)  # 이미지 띄우기
    cv2.imwrite(save_path + "prepro_" + jpg_file, bw_img)
    with open(save_path + "heatmap_" + str(file_num), "wb") as fw:
        pickle.dump(bw_img, fw)

    re_no_img, _ = img_prepro.load_single_img_resize(bw_img, 1600, 1600)


file_path = "./data/image/original_image/"  # 원본파일 경로
save_path = "./data/image/region_image/"  # 파일저장 경로
for file_num in range(52):
    image_path = file_path + str(file_num) + ".jpg"
    xml_path = file_path + str(file_num) + ".xml"
    isotropicGaussianHeatmapImage = img_prepro.make_gausian(image_path, xml_path)

    img_prepro.imshow("gaussian_map", isotropicGaussianHeatmapImage)  # 이미지 띄우기

    cv2.imwrite(
        save_path + "heatmap_" + str(file_num) + ".jpg", isotropicGaussianHeatmapImage
    )
    with open(save_path + "heatmap_" + str(file_num), "wb") as fw:
        pickle.dump(isotropicGaussianHeatmapImage, fw)
