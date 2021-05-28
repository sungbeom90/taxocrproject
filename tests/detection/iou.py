
#IOU check
# IOU 활용법
# IOU >= 0.8 일시, 해당 case는 TP(True Positive)
# precision == 전체 예측 갯수 중 TP의 갯수
# recall == 전체 정답지 갯수 중 TP의 갯수
# 비교방법 == 예측치 하나와 전체 정답지 갯수를 비교한 뒤, IOU가 0.8 이상이 나오면 break 하고 TP로 설정

def IoU(predict, answer):
    # box = (x1, y1, x2, y2)
    predict_area = (predict['xmax'] - predict['xmin'] + 1) * (predict['ymax'] - predict['ymin'] + 1)
    answer_area = (predict['xmax'] - predict['xmin'] + 1) * (predict['ymax'] - predict['ymin'] + 1)
    
    # obtain x1, y1, x2, y2 of the intersection
    inter_xmin = max(predict['xmin'], answer['xmin'])
    inter_ymin = max(predict['ymin'], answer['ymin'])
    inter_xmax = min(predict['xmax'], answer['xmax'])
    inter_ymax = min(predict['ymax'], answer['ymax'])

    # compute the width and height of the intersection
    inter_w = max(0, inter_xmax - inter_xmin + 1)
    inter_h = max(0, inter_ymax - inter_ymin + 1)

    inter_area = inter_w * inter_h
    iou = inter_area / (predict_area + answer_area - inter_area)
    return iou

def TP_check(predict_list, answer_list):
    #예측 박스와 전체 정답 박스를 비교 하되, iou가 0.8 이상이 나오면 break 후 TP 갯수로 출력
    tp_case = 0
    for predict in predict_list:
        for answer in answer_list:
            iou = IoU(predict, answer)
            if iou >= 0.8 :
                tp_case += 1
                break
            else :
                pass
    precision = tp_case / len(predict_list)
    recall =  tp_case / len(answer_list)
    return precision, recall
