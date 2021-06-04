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
        f.save("./static/image/" + secure_filename(f.filename))
        upfile_address = "image/" + secure_filename(f.filename)
        upload_file_list.append(upfile_address)
        print(upfile_address)
        # 업로드된 파일명
        return redirect(url_for("con_base"))


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


# if __name__ == "__main__":
#     app.run(debug=True)
# from app import app
