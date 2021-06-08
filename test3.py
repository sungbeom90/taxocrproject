from tests.helpers import load_data
from tests.helpers import img_prepro
import os
import cv2
import numpy as np
import pickle


# 훈련 데이터 불러오기
file_path = "./data/image/original_image/"  # 원본파일 경로
save_path = "./data/image/prepro_image/"  # 파일저장 경로
file_list_jpg = load_data.load_images(file_path, ".jpg")  # 원본파일 이름 목록 가져오기

print(file_list_jpg)


for file_num in range(52):
    jpg_file_name = file_path + str(file_num) + ".jpg"  # 파일 경로
    print("preprossing...")
    bw_np_img = img_prepro.detection_preprocess(jpg_file_name)
    print(type(bw_np_img))

    img_prepro.imshow("bw_np_img", bw_np_img)  # 이미지 띄우기

    cv2.imwrite(save_path + "prepro_" + str(file_num) + ".jpg", bw_np_img)
    with open(save_path + "prepro_" + str(file_num), "wb") as fw:
        pickle.dump(bw_np_img, fw)

    re_no_img, _ = img_prepro.load_single_img_resize(bw_np_img, 1600, 1600)


file_path = "./data/image/original_image/"  # 원본파일 경로
save_path = "./data/image/region_image/"  # 파일저장 경로
for file_num in range(52):
    image_path = file_path + str(file_num) + ".jpg"
    xml_path = file_path + str(file_num) + ".xml"
    re_np_img = img_prepro.make_gausian(image_path, xml_path)
    print(type(region_np))

    # isotropicGaussianHeatmapImage = cv2.applyColorMap(
    #         np.uint8(background), cv2.COLORMAP_BONE
    #     )  # 배경 설정에 찾아볼것

    img_prepro.imshow("gaussian_map", re_np_img)  # 이미지 띄우기

    cv2.imwrite(save_path + "region_" + str(file_num) + ".jpg", re_np_img)
    with open(save_path + "region_" + str(file_num), "wb") as fw:
        pickle.dump(re_np_img, fw)


# file_path = "./data/image/prepro_image/"  # 원본파일 경로

# with open(file_path + "prepro_1", "rb") as fr:
#     data = pickle.load(fr)

# print(data)
# print(type(data))
# print(data.shape)

# file_path = "./data/image/region_image/"  # 원본파일 경로

# with open(file_path + "heatmap_0", "rb") as fr:
#     data = pickle.load(fr)
