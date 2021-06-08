from PIL import Image
import numpy as np
import cv2


def load_single_img_resize(image_route, width: int, height: int):
    image_data = []
    im = Image.open(image_route)
    im = im.convert("L")
    im = detection_preprocess(im)

    img_width, img_height = im.size

    read = np.array(im.resize((width, height)), np.float32) / 255

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


def detection_preprocess(image: Image):
    bgrImage = np.array(image, dtype=np.uint8)
    print(bgrImage.shape)

    gray = bgrImage

    b_w = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    imshow("THRESH_BINARY_INV", b_w)  # 텍스트와 표 그리드를 백, 나머지 배경을 흑

    contours, _ = cv2.findContours(b_w, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(b_w, contours, -1, (255), 3)

    new_img = b_w

    h, w = new_img.shape[:2]

    horizontal_img = new_img
    vertical_img = new_img

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    horizontal_img = cv2.erode(horizontal_img, horizontal_kernel, iterations=1)

    imshow("y100 erode", horizontal_img)  # (100,1) 커널로 침식하여 가로줄 찾기

    horizontal_img = cv2.dilate(horizontal_img, horizontal_kernel, iterations=1)

    imshow("y100 dilate", horizontal_img)  # (100,1) 커널로 팽창

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    vertical_img = cv2.erode(vertical_img, vertical_kernel, iterations=1)

    imshow("x100 erode", vertical_img)  # (1,100) 커널로 침식하여 세로줄 찾기

    vertical_img = cv2.dilate(vertical_img, vertical_kernel, iterations=1)

    imshow("x100 dilate", vertical_img)  # (1,100) 커널로 팽창

    mask_img = horizontal_img + vertical_img

    imshow("img brodcasting", mask_img)  # 가로 세로줄 합쳐 표 얻기 (표는 백색)

    b_w2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    imshow("THRESH_BINARY", b_w2)  # 텍스트와 표 그리드를 흑, 배경을 백

    re_img = np.bitwise_or(b_w2, mask_img)

    imshow("THRESH_BINARY", re_img)  # 비트 or 연산하여 표 제거 (흑 + 백 = 백색)

    re2_img = cv2.threshold(re_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    imshow("THRESH_BINARY_re", re2_img)  # 표가 제거된 이미지를 흑백 반전하여 텍스트를 백, 배경을 흑으로

    return Image.fromarray(re2_img)


def imshow(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
