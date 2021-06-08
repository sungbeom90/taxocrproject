from PIL import Image
import numpy as np
import cv2

# 이미지 사이즈 조절 및 정규화 함수
def load_single_img_resize(img, width: int, height: int):
    image_data = []

    img_width, img_height = img.size

    read = np.array(img.resize((width, height)), np.float32) / 255

    ratio = (float(width / 2 / img_width), float(height / 2 / img_height))

    size_data = ratio

    img_arr = np.ndarray((width, height, 1), np.float32)

    pads = ((0, 0), (0, 0))

    for i in [0]:
        x = read[:, :]
        pad = np.pad(x, pads, "constant", constant_values=1)
        pad = np.resize(pad, (width, height))
        img_arr[:, :, 0] = pad

    image_data.append(img_arr)

    return img_arr, size_data


# 이미지 표 제거하는 함수
def detection_preprocess(image_route):
    img = Image.open(image_route)  # 경로에 있는 이미지 열기
    img_g = img.convert("L")  # 이미지 회색조로 변환
    img_g_np = np.array(img_g, dtype=np.uint8)  # 이미지 넘파이 uint8 타입으로 변환

    img_bw_np = cv2.threshold(
        img_g_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[
        1
    ]  # 텍스트와 표 그리드를 백, 나머지 배경을 흑
    imshow("THRESH_BINARY_INV", img_bw_np)  # 미리보기

    # contours, _ = cv2.findContours(img_bw_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # 경계 단위 묶기 (표그리기)
    # cv2.drawContours(img_bw_np, contours, -1, (255), 3)

    h, w = img_bw_np.shape[:2]
    horizontal_img = img_bw_np
    vertical_img = img_bw_np

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    horizontal_img = cv2.erode(
        horizontal_img, horizontal_kernel, iterations=1
    )  # (100,1) 커널로 침식하여 가로줄 찾기
    imshow("y100 erode", horizontal_img)  # 미리보기

    horizontal_img = cv2.dilate(
        horizontal_img, horizontal_kernel, iterations=1
    )  # (100,1) 커널로 팽창
    imshow("y100 dilate", horizontal_img)  # 미리보기

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    vertical_img = cv2.erode(
        vertical_img, vertical_kernel, iterations=1
    )  # (1,100) 커널로 침식하여 세로줄 찾기
    imshow("x100 erode", vertical_img)  # 미리보기

    vertical_img = cv2.dilate(
        vertical_img, vertical_kernel, iterations=1
    )  # (1,100) 커널로 팽창
    imshow("x100 dilate", vertical_img)  # 미리보기

    mask_img = horizontal_img + vertical_img  # 가로 세로줄 합쳐 표 얻기 (표는 백색)
    imshow("img brodcasting", mask_img)  # 미리보기

    img_wb_np = cv2.threshold(img_g_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[
        1
    ]  #  텍스트와 표 그리드를 흑, 배경을 백
    imshow("THRESH_BINARY", img_wb_np)  # 미리보기

    re_img = np.bitwise_or(img_wb_np, mask_img)  # 비트 or 연산하여 표 제거 (흑 + 백 = 백색)
    imshow("THRESH_BINARY", re_img)  # 미리보기

    re2_img = cv2.threshold(re_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[
        1
    ]  # 표가 제거된 이미지를 흑백 반전하여 텍스트를 백, 배경을 흑으로

    imshow("THRESH_BINARY_re", re2_img)  # 미리보기

    return Image.fromarray(re2_img)


def imshow(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
