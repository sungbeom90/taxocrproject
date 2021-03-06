import cv2
import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageFont
import copy
from m_model.craft_model import Craft
from m_model.recog_model import act_model_load_LSTM
from tensorflow.keras.layers import Input
import statistics


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


def recog_pre_process(or_image, word_box):

    crop_images = []
    test_img = []
    index = 0

    # ?????? ????????? ????????? ?????? ?????????
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
# ????????? ???????????? ??????
def load_single_img_resize(image_route, width: int, height: int):
    image_data = []
    im = Image.open(image_route)
    im = im.convert("L")
    im = detection_preprocess(im)  # ????????? ?????????

    img_width, img_height = im.size  # ?????? ????????? ????????? ??????

    read = np.array(im.resize((width, height)), np.float32) / 255  # ????????? 1600?????? ????????????

    ratio = (float(width / 2 / img_width), float(height / 2 / img_height))  # ????????????

    size_data = ratio  # ?????? ??????

    img_arr = np.ndarray((width, height, 1), np.float32)

    pads = ((0, 0), (0, 0))

    for i in [0]:
        x = read[:, :]
        pad = np.pad(x, pads, "constant", constant_values=1)
        pad = np.resize(pad, (width, height))
        img_arr[:, :, 0] = pad

    image_data.append(img_arr)

    return img_arr, size_data


# ????????? ????????? ?????? ?????? ??????
def pred_detection(img_route, model_weight, size):
    model = Craft()  # ????????? ?????? ??????
    model.load(model_weight)  # ????????? ????????? ??????

    test_data, _ = load_single_img_resize(img_route, size, size)  # ??????????????? ????????????
    orig_image = copy.deepcopy(test_data)  # ??????????????? ??????

    pred_map = model.predict(np.array([test_data], np.float32))  # ?????? ??????

    boxes = box_from_map(pred_map[0] * 255)  # ????????? ??????
    word_box = tax_serialization(test_data, boxes)  # ????????? ?????????(????????????)

    print("=====================")

    img = test_data

    # ???????????? ???????????? ??????
    dump_params = {
        "image": np.array(np.resize(img, (size, size)) * 255, np.uint8),
        "width": size,
        "height": size,
    }
    _, image = box_on_image(dump_params, word_box)  # ????????? ??? ?????? ??? ?????? ?????????

    or_image = np.array(orig_image * 255, np.uint8)  # ?????? ????????? ?????? ??????

    return or_image, image, word_box


# ????????? ??????????????? ?????? ?????? ??????
def pred_recognition(model_recog, char_list, or_image, word_box):
    test_image = recog_pre_process(or_image, word_box)  # ?????? ?????? ????????? ?????????
    print(test_image.shape)

    model_input = Input(shape=(32, 256, 1))

    inputs, outputs, act_model = act_model_load_LSTM(
        char_list, model_input
    )  # ??????????????? ?????? ?????? ????????????(?????????)??? ?????? ???????????? ??????????????? ??????

    act_model.load_weights(model_recog)  # ?????? ????????? ??????

    prediction = act_model.predict([test_image])  # ??????????????? ?????? ??????

    word_list = word_box
    text_list = []
    score_list = []
    score_index = []

    for index in range(len(test_image)):
        temp_loc = []
        temp_score = []
        text_temp = []

        temp = prediction[index]  # ??????????????? ?????? ?????? ?????????

        for temp_index in range(len(temp)):  # ???????????? ??????(?????????) ?????????
            if np.argmax(temp[temp_index]) == len(char_list):  # ?????? ????????? ????????? '-' ??? ??????
                pass
            else:
                if (temp_index > 0) and (np.argmax(temp[temp_index])) == (
                    np.argmax(temp[temp_index - 1])
                ):  # ?????? ?????? ??? ????????? ????????? ????????? ?????? (ex : aa)
                    pass
                else:  # ????????? ???????????? ????????? ???????????? ????????? ??????
                    temp_loc.append(
                        np.argmax(temp[temp_index])
                    )  # ????????????(?????????) ??? ????????? ??????(?????????) ????????? ?????? ??? ?????? ????????? ??????
                    temp_score.append(
                        temp[temp_index][np.argmax(temp[temp_index])]
                    )  # ????????????(?????????) ??? ????????? ??????(?????????) ????????? ?????? ??? ?????? ??? ??????

        score_index.append(temp_loc)

        if temp_score == []:  # ??? ????????? ????????? ?????? ????????? ?????? ?????? ??????
            temp_score = [0.5]
            temp_loc = [len(temp)]

        score_list.append(
            statistics.mean(temp_score)
        )  # ??????(?????????)??? ?????? ???????????? ???????????? ????????? ?????? ??????????????? ??????

        for text_index in range(len(temp_loc)):  # ??? ??????(?????????) ???????????? ?????? ????????? ?????? ??????
            text_temp.append(char_list[temp_loc[text_index]])

        string_text_temp = "".join(text_temp)  # ?????? ???????????? ????????? ??????
        text_list.append(string_text_temp)
        print(text_list[index], score_list[index])

    # ?????? ??????, ?????? ????????? ??????, ?????? ????????? ?????? ??????
    print(
        "len(text_list) : {} len(score_list) : {} len(word_list) : {}".format(
            len(text_list), len(score_list), len(word_list)
        )
    )
    return text_list, score_list, word_list


def test_logic(text_list, score_list, word_list):
    word_spot_dict = {
        "t_bill": {
            "b_id": {"text": [], "score": [], "location": (982, 126, 1524, 216)},
            "b_date": {"text": [], "score": [], "location": (76, 736, 258, 816)},
            "b_mr": {"text": [], "score": [], "location": (802, 738, 982, 820)},
            "b_etc": {"text": [], "score": [], "location": (982, 736, 1526, 818)},
            "b_cost_total": {
                "text": [],
                "score": [],
                "location": (76, 1274, 322, 1350),
            },
            "b_cost_sup": {"text": [], "score": [], "location": (258, 740, 530, 816)},
            "b_cost_tax": {"text": [], "score": [], "location": (528, 738, 802, 814)},
            "b_cost_cash": {
                "text": [],
                "score": [],
                "location": (322, 1274, 540, 1350),
            },
            "b_cost_check": {
                "text": [],
                "score": [],
                "location": (540, 1274, 760, 1350),
            },
            "b_cost_note": {
                "text": [],
                "score": [],
                "location": (760, 1274, 980, 1350),
            },
            "b_cost_credit": {
                "text": [],
                "score": [],
                "location": (980, 1274, 1200, 1350),
            },
        },
        "t_provider": {
            "p_id": {"text": [], "score": [], "location": (258, 216, 510, 300)},
            "p_corp_num": {"text": [], "score": [], "location": (618, 216, 800, 300)},
            "p_corp_name": {"text": [], "score": [], "location": (258, 300, 512, 384)},
            "p_ceo_name": {"text": [], "score": [], "location": (620, 300, 800, 384)},
            "p_add": {"text": [], "score": [], "location": (258, 382, 800, 466)},
            "p_stat": {"text": [], "score": [], "location": (266, 460, 404, 560)},
            "p_type": {"text": [], "score": [], "location": (508, 472, 800, 542)},
            "p_email": {"text": [], "score": [], "location": (256, 542, 800, 668)},
        },
        "t_item_1": {
            "i_month": {"text": [], "score": [], "location": (76, 884, 130, 966)},
            "i_day": {"text": [], "score": [], "location": (130, 884, 186, 966)},
            "i_name": {"text": [], "score": [], "location": (184, 884, 550, 966)},
            "i_stand": {"text": [], "score": [], "location": (545, 884, 676, 966)},
            "i_quan": {"text": [], "score": [], "location": (676, 884, 800, 966)},
            "i_unit": {"text": [], "score": [], "location": (800, 884, 980, 966)},
            "i_sup": {"text": [], "score": [], "location": (980, 884, 1200, 966)},
            "i_tax": {"text": [], "score": [], "location": (1200, 884, 1380, 966)},
            "i_etc": {"text": [], "score": [], "location": (1380, 884, 1522, 966)},
        },
        "t_item_2": {
            "i_month": {"text": [], "score": [], "location": (76, 966, 130, 1024)},
            "i_day": {"text": [], "score": [], "location": (130, 966, 186, 1024)},
            "i_name": {"text": [], "score": [], "location": (186, 966, 548, 1024)},
            "i_stand": {"text": [], "score": [], "location": (548, 966, 672, 1024)},
            "i_quan": {"text": [], "score": [], "location": (672, 966, 802, 1024)},
            "i_unit": {"text": [], "score": [], "location": (802, 966, 982, 1024)},
            "i_sup": {"text": [], "score": [], "location": (982, 966, 1200, 1024)},
            "i_tax": {"text": [], "score": [], "location": (1200, 966, 1380, 1024)},
            "i_etc": {"text": [], "score": [], "location": (1380, 966, 1522, 1024)},
        },
        "t_item_3": {
            "i_month": {"text": [], "score": [], "location": (76, 1024, 130, 1124)},
            "i_day": {"text": [], "score": [], "location": (130, 1024, 186, 1124)},
            "i_name": {"text": [], "score": [], "location": (186, 1024, 546, 1124)},
            "i_stand": {"text": [], "score": [], "location": (546, 1024, 672, 1124)},
            "i_quan": {"text": [], "score": [], "location": (672, 1024, 802, 1124)},
            "i_unit": {"text": [], "score": [], "location": (802, 1024, 982, 1124)},
            "i_sup": {"text": [], "score": [], "location": (982, 1024, 1200, 1124)},
            "i_tax": {"text": [], "score": [], "location": (1200, 1024, 1380, 1124)},
            "i_etc": {"text": [], "score": [], "location": (1380, 1024, 1522, 1124)},
        },
        "t_item_4": {
            "i_month": {"text": [], "score": [], "location": (76, 1124, 130, 1200)},
            "i_day": {"text": [], "score": [], "location": (130, 1124, 186, 1200)},
            "i_name": {"text": [], "score": [], "location": (186, 1124, 546, 1200)},
            "i_stand": {"text": [], "score": [], "location": (546, 1124, 672, 1200)},
            "i_quan": {"text": [], "score": [], "location": (672, 1124, 802, 1200)},
            "i_unit": {"text": [], "score": [], "location": (802, 1124, 982, 1200)},
            "i_sup": {"text": [], "score": [], "location": (982, 1124, 1200, 1200)},
            "i_tax": {"text": [], "score": [], "location": (1200, 1124, 1380, 1200)},
            "i_etc": {"text": [], "score": [], "location": (1380, 1124, 1522, 1200)},
        },
    }
    for index in range(len(text_list)):
        print("{}?????? ????????? : {} ?????? ?????????".format(index, text_list[index]))
        flag = False
        for table_key, table_value in word_spot_dict.items():
            for column_key, column_value in table_value.items():
                # print(
                #     "[{} ?????? ?????? : {}] =? [{} ?????? : {}]".format(
                #         text_list[index],
                #         word_list[index],
                #         column_key,
                #         column_value["location"],
                #     )
                # )
                if find_position(column_value["location"], word_list[index]):
                    word_spot_dict[table_key][column_key]["text"].append(
                        text_list[index]
                    )
                    word_spot_dict[table_key][column_key]["score"].append(
                        score_list[index]
                    )
                    print(
                        "{}?????? ????????? : {}, {} table, {} column??? ?????????".format(
                            index, text_list[index], table_key, column_key
                        )
                    )
                    flag = True
                    break
                if flag:
                    break
            if flag:
                break

    for table_key, table_value in word_spot_dict.items():
        for column_key, column_value in table_value.items():
            word_spot_dict[table_key][column_key]["text"] = "".join(
                column_value["text"]
            )
            word_spot_dict[table_key][column_key]["score"] = 0.98
            print(column_value["text"])
            # statistics.mean(column_value["score"])

    return word_spot_dict


def find_position(target_location, word_location):
    target_xmin, target_ymin, target_xmax, target_ymax = target_location
    word_xmin, word_ymin, word_xmax, word_ymax = word_location

    if (
        (target_xmin <= word_xmin)
        and (target_xmax >= word_xmax)
        and (target_ymin <= word_ymin)
        and (target_ymax >= word_ymax)
    ):
        return True

    else:
        return False
