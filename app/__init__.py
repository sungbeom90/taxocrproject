import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
import os
from werkzeug.utils import secure_filename
from PIL import Image
from flask import (
    Flask,
    json,
    render_template,
    redirect,
    url_for,
    request,
    jsonify,
)

from app import mod_dbconn
from app import ocr_manage as om
from m_model import Define_Class
from m_model.helpers import (
    box_from_map,
    box_on_image,
    tax_serialization,
    detection_preprocess,
    recog_pre_process,
    load_single_img_resize,
    pred_detection,
    pred_recognition,
    test_logic,
)


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


@app.route("/check_done", methods=["POST"])
def check_done():
    args_dict = request.form.to_dict()
    print(args_dict)
    p_id = {"p_id": args_dict["p_id"]}
    t_provider = {
        "p_id": args_dict["p_id"],
        "p_corp_num": args_dict["p_corp_num"],
        "p_corp_name": args_dict["p_corp_name"],
        "p_ceo_name": args_dict["p_ceo_name"],
        "p_add": args_dict["p_add"],
        "p_stat": args_dict["p_stat"],
        "p_type": args_dict["p_type"],
        "p_email": args_dict["p_email"],
    }
    t_bill = {
        "b_id": args_dict["b_id"],
        "b_date": args_dict["b_date"],
        "b_mr": args_dict["b_mr"],
        "b_etc": args_dict["b_etc"],
        "b_cost_total": om.cost_replace(args_dict["b_cost_total"]),
        "b_cost_sup": om.cost_replace(args_dict["b_cost_sup"]),
        "b_cost_tax": om.cost_replace(args_dict["b_cost_tax"]),
        "b_cost_cash": om.cost_replace(args_dict["b_cost_cash"]),
        "b_cost_check": om.cost_replace(args_dict["b_cost_check"]),
        "b_cost_note": om.cost_replace(args_dict["b_cost_note"]),
        "b_cost_credit": om.cost_replace(args_dict["b_cost_credit"]),
        "FK_p_id": args_dict["p_id"],
    }
    success = om.provider_exists(p_id)
    if success == 1:  # db에 공급자  존재함
        pass
    elif success == 0:  # db에 공급자 없음
        result = om.provider_insert(t_provider)  # 공급자 생성
    elif success == -1:  # 조회 요청 실패
        pass
    result = om.bill_insert(t_provider)

    return redirect(url_for("/bargraph"))


# ===================flaskr=========================


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
        upfile_address = "./app/static/image/" + secure_filename(f.filename)
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


# ---
@app.route("/supply_db")
def select_sup():
    sql = "SELECT *\
                FROM taxocr.t_provider"
    all_sup_dict = db_class.executeAll(sql)
    print(all_sup_dict)
    dataNum = len(all_sup_dict)
    return render_template("supply.html", resultData=all_sup_dict, dataNum=dataNum)


@app.route("/supply_desc", methods=["GET"])
# 상세 보기
def select_sup_desc():
    p_id = request.args.get("p_id")
    desc_dict = om.supply_desc(p_id)  # om 참고
    return render_template("supply_desc.html", desc_dict=desc_dict[0])


@app.route("/update_provider", methods=("GET", "POST"))
# 공급자 정보 수정버튼
def update_provider():
    print("수정요청접수")
    if request.method == "POST":
        args_dict = request.form.to_dict()
        print(args_dict)
        om.supply_update(args_dict, db_class)  # om 참고
        return redirect(url_for("select_sup"))


@app.route("/delete_provider", methods=("GET", "POST"))
# 공급자 정보 삭제버튼
def delete_provider():
    p_id = request.args.get("p_id")
    om.supply_delete(p_id, db_class)  # om 참고
    return redirect(url_for("select_sup"))


@app.route("/supply_insert", methods=["GET"])
def supply_insert():
    return render_template("supply_insert.html")


# ========================model==========================


@app.route("/predict")
def predict():
    # tensorflow version (1.0 -> 2.0)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    char_list = Define_Class.definition_class()  # 텍스트 클래스 종합 파일 로드

    # Text Class 확인 (생략함)
    # print("Class num = ", len(char_list))
    # print("Model CLASS")

    # 텍스트 클래스 35개씩 프린트 (생략함)
    # for temp_index in range(len(char_list)):
    #     if temp_index % 35 == 0:
    #         print(char_list[temp_index : temp_index + 35])
    #         print("\n")

    model = "./data/trained_weights/1600_pdfdata_origin_d05_decay1000_1600to1600_20210213-2203_last"  # 디텍션 모델 가중치 로드
    model_recog = "./data/trained_weights/tax_save_model_0309.hdf5"  # 리코그니션 모델 가중치 로드

    jpg_file_name = upload_file_list.pop()  # 입력 이미지 경로 로드

    print("Detecting...")  # 이미지 디텍팅 실행
    or_image, boxed_image, word_box = pred_detection(jpg_file_name, model, size=1600)

    print(len(word_box))  # 단어단위 디텍션 완성 좌표값 저장 갯수보기 (xmin, ymin, xmax, ymax) 구조
    print(word_box)

    # 이미지 시각화 (생략)
    # plt.imshow(boxed_image, 'Greys_r')
    # plt.show()
    # plt.xticks([]), plt.yticks([])
    # plt.axis('off')
    # plt.tight_layout()
    # fig = plt.gcf()

    print("Recognizing...")  # 이미지 리코그닝 실행
    text_list, score_list, word_list = pred_recognition(
        model_recog, char_list, or_image, word_box
    )

    # with open("./data/pickle/text_list.pickle", "wb") as f:
    #     pickle.dump(text_list, f, pickle.HIGHEST_PROTOCOL)

    # with open("./data/pickle/score_list.pickle", "wb") as f:
    #     pickle.dump(score_list, f, pickle.HIGHEST_PROTOCOL)

    # with open("./data/pickle/word_list.pickle", "wb") as f:
    #     pickle.dump(word_list, f, pickle.HIGHEST_PROTOCOL)

    word_spot_dict = test_logic(text_list, score_list, word_list)

    return render_template(
        "con_base.html", jpg_file_name=jpg_file_name, word_spot_dict=word_spot_dict
    )
    # return redirect(
    #     url_for(
    #         "logic", text_list=text_list, score_list=score_list, word_list=word_list
    #     )
    # )


# @app.route("/logic", methods=["GET"])
# def logic():
#     text_list = request.args.get("text_list")  # 워드 텍스트 리스트
#     score_list = request.args.get("score_list")  # 워드 확률 리스트
#     word_list = request.args.get("word_list")  # 워드 좌표 리스트

#     t_bill_b_id
#     for index in range(len(text_list)):

#         if find_position(t_bill_b_id_location, word_list[index]):
#             t_bill_b_id.append((text_list[index], score_list[index]))


#     return redirect(url_for("con_base"))
