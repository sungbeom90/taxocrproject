# with open("after(1024)/heatmap_0", "rb") as file:  # james.p 파일을 바이너리 읽기 모드(rb)로 열기
#    data = pickle.load(file)
import cv2
import numpy as np


def fun_decoding(region_score_map):
    region_score_map = np.around(region_score_map)
    region_score_map = region_score_map.astype("uint8")
    cv2.imshow("img", region_score_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # region_score_map = cv2.cvtColor(region_score_map, cv2.COLOR_BGR2GRAY)
    # image = cv2.imread("after(1024)/after_0.jpg")
    _, result = cv2.threshold(
        region_score_map, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    # grouped object의 좌표 (x, y, w, h) 및 CENTROID(무게중심) 산출
    ret, labels, stats, centriods = cv2.connectedComponentsWithStats(
        result, connectivity=4, ltype=cv2.CV_32S
    )
    # for x, y, w, h, cnt in stats:
    #     print("x:{}, y:{}, w:{}, h:{},".format(x, y, w, h))

    bounding_box_list = []
    for i, temp in enumerate(stats):
        if i == 0:
            continue
        else:
            pass
        bounding_box_dict = {}
        # 4방향 1로 초기화
        # 글자가 총 몇 개인지 알기. 히트맵_0은 0~291개
        width_left = 1
        width_right = 1
        height_top = 1
        height_bottem = 1
        # print("{}번째 텍스트 박스".format(i))
        # 실제 사용하는 것은 x, y, w, h이며,
        x, y, w, h, cnt = temp
        # 잡음을 제거한 이미지에서
        # 텍스트 상자의 좌표 값을 구함.
        box = region_score_map[y : y + h, x : x + w]
        box_shape = (h, w)  # h행 w열
        # print(box)
        # print(box_shape)
        # 텍스트의 중심점
        # np.argmax()함수가 h행w열에서 가장 큰 인덱스 값을 찾을것이다.
        # 그럼, np.unravel_index()함수가 튜플로 반환할 것임.
        x_center = np.unravel_index(np.argmax(box, axis=None), box.shape)[1]
        y_center = np.unravel_index(np.argmax(box, axis=None), box.shape)[0]
        # print("x_center:{}, y_center:{}".format(x_center, y_center))
        x1 = x + x_center
        y1 = y + y_center
        x2 = x + x_center
        y2 = y + y_center
        x3 = x + x_center
        y3 = y + y_center
        x4 = x + x_center
        y4 = y + y_center
        # 중심점~ 각 4방향에 대한, 화소값을 비교하여 영역을 확장함.
        # 중심점은 가장 큰 값이며, 멀어질수록 값이 작아짐
        # width_left : 왼쪽
        while True:
            if (
                region_score_map[y1][x1 - width_left]
                <= region_score_map[y1][x1 - width_left + 1]
            ) and (region_score_map[y1][x1 - width_left + 1] > 60):
                width_left += 1
                # print(
                #     "왼쪽 좌표[x:{}][y:{}]:화소값{}, 좌표[y:{}][x:{}]:화소값:{}".format(
                #         x1 - width_left,
                #         y1,
                #         (region_score_map[y1][x1 - width_left]),
                #         x1 - width_left + 1,
                #         y1,
                #         (region_score_map[y1][x1 - width_left + 1]),
                #     )
                # )
            else:
                xmin = x1 - width_left + 1
                # print("xmin 선택 {} ".format(xmin))
                bounding_box_dict["xmin"] = xmin
                break
        # width_right : 오른쪽
        while True:
            print("y2 :{}, x2 :{}, width_right : {}".format(y2, x2, width_right))
            if (
                region_score_map[y2][x2 + width_right]
                <= region_score_map[y2][x2 + width_right - 1]
            ) and (region_score_map[y2][x2 + width_right - 1] > 60):
                width_right += 1
                # print(
                #     "오른쪽 좌표[x:{}][y:{}]:화소값{}, 좌표[y:{}][x:{}]:화소값:{}".format(
                #         x2 + width_right,
                #         y2,
                #         (region_score_map[y2][x2 + width_right]),
                #         x2 + width_right + 1,
                #         y2,
                #         (region_score_map[y2][x2 + width_right - 1]),
                #     )
                # )
            else:
                xmax = x2 + width_right - 1
                # print("xmax 선택 {} ".format(xmax))
                bounding_box_dict["xmax"] = xmax
                break
        # height_top : 위
        while True:
            if (
                region_score_map[y3 - height_top][x3]
                <= region_score_map[y3 - height_top + 1][x3]
            ) and (region_score_map[y3 - height_top + 1][x3] > 60):
                height_top += 1
                # print(
                #     "위쪽 좌표[x:{}][y:{}]:화소값{}, 좌표[y:{}][x:{}]:화소값:{}".format(
                #         x3,
                #         y3 - height_top,
                #         (region_score_map[y3 - height_top][x3]),
                #         x3,
                #         y3 - height_top + 1,
                #         (region_score_map[y3 - height_top + 1][x3]),
                #     )
                # )
            else:
                ymin = y3 - height_top
                # print("ymin 선택 {} ".format(ymin))
                bounding_box_dict["ymin"] = ymin
                break
        # height_bottem : 아래
        while True:
            if (
                region_score_map[y4 + height_bottem][x4]
                <= region_score_map[y4 + height_bottem - 1][x4]
            ) and (region_score_map[y4 + height_bottem - 1][x4] > 60):
                height_bottem += 1
                # print(
                #     "아래쪽 좌표[x:{}][y:{}]:화소값{}, 좌표[y:{}][x:{}]:화소값:{}".format(
                #         x4,
                #         y4 + height_bottem,
                #         (region_score_map[y4 + height_bottem][x4]),
                #         x4,
                #         y4 + height_bottem - 1,
                #         (region_score_map[y4 + height_bottem - 1][x4]),
                #     )
                # )
            else:
                ymax = y4 + height_bottem
                # print("ymax 선택 {} ".format(ymax))
                bounding_box_dict["ymax"] = ymax
                break
        # print("bounding_box_dict : {}".format(bounding_box_dict))
        bounding_box_list.append(bounding_box_dict)

    return bounding_box_list

    # for a in range(len(bounding_box_list)):
    #     xmin = int(bounding_box_list[a].get("xmin"))
    #     xmax = int(bounding_box_list[a].get("xmax"))
    #     ymin = int(bounding_box_list[a].get("ymin"))
    #     ymax = int(bounding_box_list[a].get("ymax"))
    #     img = cv2.rectangle(blur, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
    # for a in range(len(bounding_box_list)):
    #     xmin = int(bounding_box_list[a].get("xmin"))
    #     xmax = int(bounding_box_list[a].get("xmax"))
    #     ymin = int(bounding_box_list[a].get("ymin"))
    #     ymax = int(bounding_box_list[a].get("ymax"))
    #     img2 = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    # cv2.imshow("img", img)
    # cv2.imshow("img2", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
