from flask import Flask, render_template, redirect, request, url_for
from app import mod_dbconn

app = Flask(__name__)
db_class = mod_dbconn.Database()

# 업로드된 파일주소가 저장되는 리스트
upload_file_list = []
app.config["UPLOAD_DIR"] = "./static/image/"


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/db")
def select():
    sql = "SELECT *\
                FROM taxocr.t_provider"
    row = db_class.executeAll(sql)
    print(row)
    return render_template("db.html", resultData=row)


@app.route("/insert_provider", methods=("GET", "POST"))
def insert_provider():
    print("공급자 등록 요청 접수됨")
    if request.method == "GET":
        return render_template("insert.html")
    if request.method == "POST":

        args = tuple(request.form.values())
        print(args)
        sql = """INSERT into taxocr.t_provider (p_id, p_corp_num, p_corp_name, p_ceo_name, p_add, p_stat, p_type, p_email)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"""
        db_class.execute(query=sql, args=args)
        db_class.commit()
        return render_template("home.html")


@app.route("/insert_bill", methods=("GET", "POST"))
def insert_bill():
    print("계산서 등록 요청 접수됨")
    if request.method == "GET":
        return render_template("insert.html")
    if request.method == "POST":
        args_dict = request.form.to_dict()
        print(args_dict)
        sql = """INSERT into taxocr.t_bill (b_id, b_date, b_mr, b_etc, b_cost_total, b_cost_sup, b_cost_tax,
                                            b_cost_cash, b_cost_check, b_cost_note, b_cost_credit, FK_p_id)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        args = (
            args_dict["b_id"],
            args_dict["b_date"],
            args_dict["b_mr"],
            args_dict["b_etc"],
            int(args_dict["b_cost_total"].replace(",", "")),
            int(args_dict["b_cost_sup"].replace(",", "")),
            int(args_dict["b_cost_tax"].replace(",", "")),
            int(args_dict["b_cost_cash"].replace(",", "")),
            int(args_dict["b_cost_check"].replace(",", "")),
            int(args_dict["b_cost_note"].replace(",", "")),
            int(args_dict["b_cost_credit"].replace(",", "")),
            args_dict["FK_p_id"],
        )
        print(args)
        db_class.execute(query=sql, args=args)
        db_class.commit()
        return render_template("home.html")


@app.route("/bargraph")
def barGraph():
    title = "bargraph"
    labels = []  # 도넛그래프 x축 : 회사명
    data = []  # 도넛그래프 y축 : 거래 금액
    data2 = []  # 라인 그래프 y축 : 거래 금액
    data3 = []  # 막대 그래프 y축 : 수단별 거래 금액

    # 연도 받아서 sql 바꿔야함
    year_ = "2010"

    # 세금 데이터----------------------------------------
    taxsql = """SELECT SUM(t_bill.b_cost_tax)
                FROM t_bill
                WHERE YEAR(t_bill.b_date) = '2010'"""
    taxrow = db_class.executeOne(taxsql)

    taxdata = int(taxrow["SUM(t_bill.b_cost_tax)"])

    print("taxdata는 -> ", taxdata)

    # Doughnut graph------------------------------------
    temp = []
    temp2 = []
    temptuple = []
    sql = """SELECT t_provider.p_corp_name as p_corp_name,
            SUM(t_bill.b_cost_total) as b_cost_total_sum
            FROM t_provider, t_bill
            WHERE t_provider.p_id = t_bill.FK_p_id AND YEAR(t_bill.b_date) = '2010'
            GROUP BY FK_p_id
            ORDER BY SUM(t_bill.b_cost_total) DESC"""
    row = db_class.executeAll(sql)
    print("fetchall row:{}".format(row))
    print("fetchall rowtype:{}".format(type(row)))
    print("fetchall rowlength:{}".format(len(row)))
    for i in row:
        temp.append(i["p_corp_name"])
        temp2.append(int(i["b_cost_total_sum"]))
    print(temp)
    print(temp2)

    # 회사명, 거래금액
    for i in range(len(temp)):
        a = (temp[i], temp2[i])
        temptuple.append(a)
    print(temptuple)

    # 회사명 뽑아오기
    for i in range(len(temptuple)):
        labels.append(temptuple[i][0])

    print(labels)

    # 거래 금액
    for i in range(len(temptuple)):
        data.append(temptuple[i][1])

    print(data)

    # Line graph---------------------------------------------
    linesql = """SELECT MONTH(t_bill.b_date), SUM(t_bill.b_cost_total)
                FROM t_bill
                WHERE YEAR(t_bill.b_date) = '2010' 
                GROUP BY MONTH(t_bill.b_date)"""
    linerow = db_class.executeAll(linesql)  # [{'b_data':'1','b_cost_total':10000}, ...]

    print(linerow)

    # dict에서 금액 정보만 빼오기
    for i in linerow:
        data2.append(int(i["SUM(t_bill.b_cost_total)"]))
    print(data2)

    # Bar graph-----------------------------------------------
    barsql = """SELECT SUM(t_bill.b_cost_total),
                SUM(t_bill.b_cost_cash),
                SUM(t_bill.b_cost_check),
                SUM(t_bill.b_cost_note),
                SUM(t_bill.b_cost_credit) 
        FROM t_bill
        WHERE YEAR(t_bill.b_date) = '2010'"""

    barrow = db_class.executeAll(
        barsql
    )  # [{'b_cost_total':50000,'b_cost_cash':50000,'b_cost_check':50000,'b_cost_note':50000, 'b_cost_credit':50000}, {}, ...]

    print(barrow)

    for i in barrow:
        data3.append(int(i["SUM(t_bill.b_cost_cash)"]))
        data3.append(int(i["SUM(t_bill.b_cost_check)"]))
        data3.append(int(i["SUM(t_bill.b_cost_note)"]))
        data3.append(int(i["SUM(t_bill.b_cost_credit)"]))
    print(data3)

    return render_template(
        "bargraph.html",
        title=title,
        taxdata=taxdata,
        labels=labels,
        data=data,
        data2=data2,
        data3=data3,
    )


# ===================flaskr=========================
from flask import Flask, json, render_template, redirect, url_for, request, jsonify
import app.ocr_manage as om
import os
from werkzeug.utils import secure_filename

# 업로드된 파일주소가 저장되는 리스트
upload_file_list = []
app.config["UPLOAD_DIR"] = "./static/image/"


@app.route("/dashboard")
def dashboard():
    # 대시보드로 이동합니다.
    # 대시보드를 조회할 수 있는 함수는 이곳에 구현하시면 됩니다.
    return render_template("dashboard.html")


@app.route("/done")
def file_print():
    # 이미지 경로 설정
    image_upload = upload_file_list.pop()
    # image_path2는 text파일로 변동 예정임
    # image_path2 = "./image/after_0.jpg"
    return render_template("service.html", image_file=image_upload)


@app.route("/uploader", methods=["GET", "POST"])
def upload_file():
    print("파일업로드 요청 접수됨")
    if request.method == "POST":
        # 파일 저장
        f = request.files["file"]
        print("file storage 내부 : ", f)
        f.save("./app/static/image/" + secure_filename(f.filename))
        upfile_address = "./static/image/" + secure_filename(f.filename)
        upload_file_list.append(upfile_address)
        print(upfile_address)
        # 업로드된 파일명
        return redirect(url_for("predict"))


@app.route("/guide", methods=["GET"])
def guide():
    return render_template("guide.html")


@app.route("/predicted_img", methods=["POST"])
def Predict_img():
    # 예측된 이미지에서 text를 출력해서 보내왔습니다.
    jsonData = request.get_json()
    data1 = str(jsonData["testkey"])
    return jsonData


@app.route("/con_base", methods=["GET"])
def con_base():
    # 업로드된 사진 출력
    image_upload = upload_file_list.pop()
    return render_template("con_base.html", image_file=image_upload)


@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == "GET":
        return render_template("home.html")
    elif request.method == "POST":
        Data = request.get_json()
        print(Data)
        Data1 = str(Data["test1"])
        print(Data1)
        return render_template("test1.html", Data1=Data1)


# ========================model==========================
from m_model.craft_model import Craft
from m_model.helpers import (
    box_from_map,
    box_on_image,
    tax_serialization,
    detection_preprocess,
)
from m_model.helpers import recog_pre_process
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import tensorflow as tf
import m_model.Define_Class as Define_Class
import m_model.recog_model as recog_model
from tensorflow.keras.layers import Input
import statistics

# 이미지 리사이즈 함수
def load_single_img_resize(image_route, width: int, height: int):
    image_data = []
    im = Image.open(image_route)
    im = im.convert("L")
    im = detection_preprocess(im)  # 이미지 전처리

    img_width, img_height = im.size  # 실제 이미지 사이즈 저장

    read = np.array(im.resize((width, height)), np.float32) / 255  # 이미지 1600으로 리사이즈

    ratio = (float(width / 2 / img_width), float(height / 2 / img_height))

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
def pred_test(img_route, model_weight, size):
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


@app.route("/predict")
def predict():
    # tensorflow version (1.0 -> 2.0)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    char_list = Define_Class.definition_class()  # 텍스트 클래스 종합 파일 로드

    # Text Class 확인
    print("Class num = ", len(char_list))
    print("Model CLASS")

    # 텍스트 클래스 35개씩 프린트
    for temp_index in range(len(char_list)):
        if temp_index % 35 == 0:
            print(char_list[temp_index : temp_index + 35])
            print("\n")

    model = "./data/trained_weights/1600_pdfdata_origin_d05_decay1000_1600to1600_20210213-2203_last"  # 디텍션 모델 가중치 로드
    model_recog = "./data/trained_weights/tax_save_model_0309.hdf5"  # 리코그니션 모델 가중치 로드

    jpg_file_name = upload_file_list[-1]  # 입력 이미지 경로 로드

    print("Detecting...")
    or_image, boxed_image, word_box = pred_test(jpg_file_name, model, size=1600)

    print(len(word_box))
    print(word_box)

    # plt.imshow(boxed_image, 'Greys_r')
    # plt.show()
    # plt.xticks([]), plt.yticks([])
    # plt.axis('off')
    # plt.tight_layout()
    # fig = plt.gcf()

    crop_image = []

    for index in range(len(word_box)):
        crop_image.append(
            or_image[
                word_box[index][1] : word_box[index][3],
                word_box[index][0] : word_box[index][2],
            ]
        )

    print("Recognizing...")
    test_image = recog_pre_process(crop_image)
    print(test_image.shape)

    model_input = Input(shape=(32, 256, 1))

    inputs, outputs, act_model = recog_model.act_model_load_LSTM(char_list, model_input)

    act_model.load_weights(model_recog)

    prediction = act_model.predict([test_image])

    word_list = word_box
    text_list = []
    score_list = []
    score_index = []

    for index in range(len(test_image)):
        temp_loc = []
        temp_score = []
        text_temp = []

        temp = prediction[index]

        for temp_index in range(len(temp)):
            if np.argmax(temp[temp_index]) == len(char_list):
                pass
            else:
                if (temp_index > 0) and (np.argmax(temp[temp_index])) == (
                    np.argmax(temp[temp_index - 1])
                ):
                    pass
                else:
                    temp_loc.append(np.argmax(temp[temp_index]))
                    temp_score.append(temp[temp_index][np.argmax(temp[temp_index])])

        score_index.append(temp_loc)

        if temp_score == []:
            temp_score = [0.5]
            temp_loc = [len(temp)]

        score_list.append(statistics.mean(temp_score))

        for text_index in range(len(temp_loc)):
            text_temp.append(char_list[temp_loc[text_index]])

        string_text_temp = "".join(text_temp)
        text_list.append(string_text_temp)
        print(text_list[index], score_list[index])

    print(
        "len(test_list) : {} len(score_list) : {} len(word_list) : {}".format(
            len(test_list), len(score_list), len(word_list)
        )
    )

    return redirect(
        url_for(
            "logic", test_list=test_list, score_list=score_list, word_list=word_list
        )
    )


@app.route("/logic", methods=["GET"])
def logic():
    test_list = request.args.get("test_list")  # 워드 텍스트 리스트
    score_list = request.args.get("score_list")  # 워드 확률 리스트
    word_list = request.args.get("word_list")  # 워드 좌표 리스트

    t_bill_b_id
    for index in range(len(test_list)):

        if find_position(t_bill_b_id_location, word_list[index]):
            t_bill_b_id.append((test_list[index], score_list[index]))

    return redirect(url_for("con_base"))


def find_position(target_location, word_location):
    target_xmin, target_xman, target_ymin, target_ymax = tartget_location
    word_xmin, word_xmax, word_ymin, word_ymax = word_location

    if (
        target_xmin <= word_xmin
        and target_xmax >= word_xmax
        and target_ymin <= word_ymin
        and target_ymax >= word_ymax
    ):
        return True

    else:
        return False
