import cv2
import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageFont
import copy
from m_model.craft_model import Craft
from tensorflow.keras.layers import Input
import m_model.recog_model as recog_model


def box_from_map(heat_map):
    padding = 0
    center_min = 10

    heat_map = np.array(heat_map).astype(np.uint8)
    ret, thresh = cv2.threshold(heat_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    connectivity = 4
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)

    stats = output[2]
    centroid = output[3]
    centers = []

    for stat, center in zip(stats, centroid):
        box = heat_map[stat[1] : stat[1] + stat[3], stat[0] : stat[0] + stat[2]]
        x = np.unravel_index(np.argmax(box, axis=None), box.shape)[0] + stat[1]
        y = np.unravel_index(np.argmax(box, axis=None), box.shape)[1] + stat[0]
        if heat_map[x][y] > center_min:
            centers.append((x, y))

    boxes = make_boxes(centers, heat_map, padding)

    return boxes


def make_boxes(centers, heat_map, padding):
    boxes = []

    for center in centers:

        cur_x = center[1]
        cur_y = center[0]

        width_left = 1
        width_right = 1
        height_up = 1
        height_down = 1

        threshold = 5

        while True:
            try:
                if (
                    heat_map[cur_y][cur_x - width_left]
                    <= heat_map[cur_y][cur_x - width_left + 1]
                    and heat_map[cur_y][cur_x - width_left + 1] > threshold
                ):
                    width_left += 1
                else:
                    break
            except IndexError:
                break

        while True:
            try:
                if (
                    heat_map[cur_y][cur_x + width_right]
                    <= heat_map[cur_y][cur_x + width_right - 1]
                    and heat_map[cur_y][cur_x + width_right - 1] > threshold
                ):
                    width_right += 1
                else:
                    break
            except IndexError:
                break

        while True:
            try:
                if (
                    heat_map[cur_y - height_up][cur_x]
                    <= heat_map[cur_y - height_up + 1][cur_x]
                    and heat_map[cur_y - height_up + 1][cur_x] > threshold
                ):
                    height_up += 1
                else:
                    break
            except IndexError:
                break

        while True:
            try:
                if (
                    heat_map[cur_y + height_down][cur_x]
                    <= heat_map[cur_y + height_down - 1][cur_x]
                    and heat_map[cur_y + height_down - 1][cur_x] > threshold
                ):
                    height_down += 1
                else:
                    break
            except IndexError:
                break

        height = max(height_down, height_up) + padding
        width = max(width_left, width_right) + padding
        if height > 0 and width > 0:
            boxes.append(
                [
                    cur_x - width_right,
                    cur_y - height_up,
                    cur_x + width_right,
                    cur_y + height_down,
                ]
            )

    return boxes


def box_on_image(parameters, boxes):
    image = parameters["image"]
    width = parameters["width"]
    height = parameters["height"]

    image = Image.fromarray(image).resize((int(width), int(height)))
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype("./m_model/font/NanumGothic.ttf", 20)

    for i, box in enumerate(boxes):
        draw.rectangle(box, outline="blue")
        draw.text((box[2] + 5, box[1] + 5), str(i), font=font)

    image = image.resize((width, height))
    return parameters, image


def detection_preprocess(image: Image):
    bgrImage = np.array(image, dtype=np.uint8)
    print(bgrImage.shape)

    gray = bgrImage

    b_w = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(b_w, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(b_w, contours, -1, (255), 3)

    new_img = b_w

    h, w = new_img.shape[:2]

    horizontal_img = new_img
    vertical_img = new_img

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    horizontal_img = cv2.erode(horizontal_img, horizontal_kernel, iterations=1)
    horizontal_img = cv2.dilate(horizontal_img, horizontal_kernel, iterations=1)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    vertical_img = cv2.erode(vertical_img, vertical_kernel, iterations=1)
    vertical_img = cv2.dilate(vertical_img, vertical_kernel, iterations=1)

    mask_img = horizontal_img + vertical_img

    b_w2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    re_img = np.bitwise_or(b_w2, mask_img)

    return Image.fromarray(re_img)


def find_box(coordinate):
    tmp_cor = []
    if coordinate[0][0] < coordinate[0][2]:
        for x1, y1, x2, y2 in coordinate:
            tmp_cor.append([x1, y1, x2 - x1, y2 - y1])
        coordinate = tmp_cor
    start_x = coordinate[0][0]
    start_y = coordinate[0][1]
    end_x = coordinate[0][0]
    end_y = coordinate[0][1]
    for x, y, w, h in coordinate:
        if end_x * 2 > x:
            end_x = max(end_x, x + w)
            end_y = max(end_y, y + h)
        if start_x > x:
            start_x = x

    return int(start_x), int(start_y), int(end_x), int(end_y)


def tax_serialization(test_img, coors):

    serialized_line = []

    tmp_line = []

    ycoors = sorted(coors, key=lambda coors: coors[1])

    save_ycoors = ycoors

    for co_index in range(len(ycoors)):
        y_temp_list_min = [ycoors[co_index][1]]
        y_temp_list_max = [ycoors[co_index][3]]

        y_sort_thres = int((ycoors[co_index][3] - ycoors[co_index][1]) / 5)

        for co_index2 in range(len(ycoors)):
            if abs(ycoors[co_index][1] - ycoors[co_index2][1]) <= y_sort_thres:
                y_temp_list_min.append(ycoors[co_index2][1])
                y_temp_list_max.append(ycoors[co_index2][3])

        save_ycoors[co_index][1] = min(y_temp_list_min)
        save_ycoors[co_index][3] = max(y_temp_list_max)

    xycoors = sorted(
        save_ycoors, key=lambda save_ycoors: (save_ycoors[1], save_ycoors[0])
    )

    for i, coor in enumerate(xycoors):
        if i == (len(xycoors) - 1):
            if len(tmp_line) == 0:
                tmp_line.append(xycoors[i])
            else:
                x, y, w, h = find_box(tmp_line)
                cv2.rectangle(test_img, (x, y), (w, h), (0, 255, 0), 2)
                serialized_line.append([x, y, w, h])
            break

        x_thres = (
            int(
                (
                    abs(xycoors[i][2] - xycoors[i][0])
                    + abs(xycoors[i + 1][2] - xycoors[i + 1][0])
                )
                * 0.5
            )
            + 1
        )
        y_thres = (
            int(
                (
                    abs(xycoors[i][3] - xycoors[i][1])
                    + abs(xycoors[i + 1][3] - xycoors[i + 1][1])
                )
                * 0.1
            )
            + 1
        )

        if len(tmp_line) == 0:
            tmp_line.append(xycoors[i])

        if abs(xycoors[i][3] - xycoors[i + 1][3]) <= y_thres:
            if abs(xycoors[i][2] - xycoors[i + 1][0]) <= x_thres:
                tmp_line.append(xycoors[i + 1])
            else:
                if len(tmp_line) == 0:
                    tmp_line.append(xycoors[i])
                x, y, w, h = find_box(tmp_line)
                cv2.rectangle(test_img, (x, y), (w, h), (0, 255, 0), 2)
                serialized_line.append([x, y, w, h])
                tmp_line = []
        else:
            if len(tmp_line) == 0:
                tmp_line.append(xycoors[i])
            x, y, w, h = find_box(tmp_line)
            cv2.rectangle(test_img, (x, y), (w, h), (0, 255, 0), 2)
            serialized_line.append([x, y, w, h])
            tmp_line = []

    return serialized_line


def recog_pre_process(word_box):

    crop_images = []
    test_img = []
    index = 0

    # 단어 좌표를 이용한 단어 자르기
    for index in range(len(word_box)):
        crop_images.append(
            or_image[
                word_box[index][1] : word_box[index][3],
                word_box[index][0] : word_box[index][2],
            ]
        )

    for crop_image in crop_images:

        crop_image_resizing = cv2.resize(
            crop_image, dsize=(256, 32), interpolation=cv2.INTER_AREA
        )
        crop_image_dim = np.expand_dims(crop_image_resizing, axis=2)

        crop_image_resizing = crop_image_resizing / 255.0

        test_img.append(crop_image_resizing)

    test_img = np.array(test_img)

    return test_img


# ================[move] test.py -> helpers.py  ======================
# 이미지 리사이즈 함수
def load_single_img_resize(image_route, width: int, height: int):
    image_data = []
    im = Image.open(image_route)
    im = im.convert("L")
    im = detection_preprocess(im)  # 이미지 전처리

    img_width, img_height = im.size  # 실제 이미지 사이즈 저장

    read = np.array(im.resize((width, height)), np.float32) / 255  # 이미지 1600으로 리사이즈

    ratio = (float(width / 2 / img_width), float(height / 2 / img_height))  # 비율계산

    size_data = ratio  # 비율 저장

    img_arr = np.ndarray((width, height, 1), np.float32)

    pads = ((0, 0), (0, 0))

    for i in [0]:
        x = read[:, :]
        pad = np.pad(x, pads, "constant", constant_values=1)
        pad = np.resize(pad, (width, height))
        img_arr[:, :, 0] = pad

    image_data.append(img_arr)

    return img_arr, size_data


# 이미지 디텍션 모델 실행 함수
def pred_detection(img_route, model_weight, size):
    model = Craft()  # 디텍션 모델 생성
    model.load(model_weight)  # 디테션 가중치 주입

    test_data, _ = load_single_img_resize(img_route, size, size)  # 입력이미지 리사이즈
    orig_image = copy.deepcopy(test_data)  # 원본이미지 저장

    pred_map = model.predict(np.array([test_data], np.float32))  # 모델 예측

    boxes = box_from_map(pred_map[0] * 255)  # 화소값 적용
    word_box = tax_serialization(test_data, boxes)  # 텍스트 디코딩(박스처리)

    print("=====================")

    img = test_data

    # 파라미터 딕셔너리 선언
    dump_params = {
        "image": np.array(np.resize(img, (size, size)) * 255, np.uint8),
        "width": size,
        "height": size,
    }
    _, image = box_on_image(dump_params, word_box)  # 이미지 위 박스 및 순서 그리기

    or_image = np.array(orig_image * 255, np.uint8)  # 원본 이미지 화소 적용

    return or_image, image, word_box


def pred_recognition(model_recog, word_box):
    test_image = recog_pre_process(word_box)  # 단어 크롭 이미지 전처리
    print(test_image.shape)

    model_input = Input(shape=(32, 256, 1))

    inputs, outputs, act_model = recog_model.act_model_load_LSTM(
        char_list, model_input
    )  # 리코그니션 모델 생성 종속변수(클래스)와 입력 사이즈를 매게변수로 제공

    act_model.load_weights(model_recog)  # 모델 가중치 적용

    prediction = act_model.predict([test_image])  # 리코그니션 모델 예측

    word_list = word_box
    text_list = []
    score_list = []
    score_index = []

    for index in range(len(test_image)):
        temp_loc = []
        temp_score = []
        text_temp = []

        temp = prediction[index]  # 예측값에서 단어 하나 꺼내기

        for temp_index in range(len(temp)):  # 단어에서 활자(케릭터) 꺼내기
            if np.argmax(temp[temp_index]) == len(char_list):  # 모델 예측시 생성된 '-' 면 제거
                pass
            else:
                if (temp_index > 0) and (np.argmax(temp[temp_index])) == (
                    np.argmax(temp[temp_index - 1])
                ):  # 모델 예측 시 생성된 반복된 글자면 제거 (ex : aa)
                    pass
                else:  # 위에서 필터링된 활자만 인정하여 단어에 넣음
                    temp_loc.append(
                        np.argmax(temp[temp_index])
                    )  # 종속변수(클래스) 에 명시된 활자(케릭터) 확률중 가장 큰 확률 인덱스 저장
                    temp_score.append(
                        temp[temp_index][np.argmax(temp[temp_index])]
                    )  # 종속변수(클래스) 에 명시된 활자(케릭터) 확률중 가장 큰 확률 값 저장

        score_index.append(temp_loc)

        if temp_score == []:  # 위 조건중 패스된 경우 아래와 같은 값을 저장
            temp_score = [0.5]
            temp_loc = [len(temp)]

        score_list.append(
            statistics.mean(temp_score)
        )  # 활자(케릭터)에 대한 확률값을 평균내어 단어의 대한 확률값으로 저장

        for text_index in range(len(temp_loc)):  # 각 활자(케릭터) 인덱스로 실제 텍스트 얻어 저장
            text_temp.append(char_list[temp_loc[text_index]])

        string_text_temp = "".join(text_temp)  # 실제 텍스트를 단어로 묶기
        text_list.append(string_text_temp)
        print(text_list[index], score_list[index])

    # 단어 갯수, 단어 확률값 갯수, 단어 좌표값 갯수 보기
    print(
        "len(text_list) : {} len(score_list) : {} len(word_list) : {}".format(
            len(text_list), len(score_list), len(word_list)
        )
    )
    return text_list, score_list, word_list
