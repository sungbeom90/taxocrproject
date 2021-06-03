from flask import Flask, render_template, redirect, request, url_for
from app import mod_dbconn

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/db")
def select():
    db_class = mod_dbconn.Database()
    sql = "SELECT *\
                FROM taxocr.t_provider"
    row = db_class.executeAll(sql)
    print(row)
    return render_template("db.html", resultData=row[0])


@app.route("/insert_provider", methods=("GET", "POST"))
def insert_provider():
    print("공급자 등록 요청 접수됨")
    if request.method == "GET":
        return render_template("insert.html")

    if request.method == "POST":
        db_class = mod_dbconn.Database()
        sql = "INSERT into taxocr.t_provider (p_id, p_crop_num, p_corp_name, p_ceo_name, p_add, p_stat, p_type, p_email) *\
                VALUES (:p_id, :p_crop_num, :p_corp_name, :p_ceo_name, :p_add, :p_stat, :p_type, :p_email)"
        args = {
            "p_id": p_id,
            "p_crop_num": p_corp_num,
            "p_corp_name": p_corp_name,
            "p_ceo_name": p_ceo_name,
            "p_add": p_add,
            "p_stat": p_stat,
            "p_type": p_type,
            "p_email": p_email,
        }
        db_class.execute(sql, args)
        return render_template("home.html")


@app.route("/")
def printHello():
    return "Hello World - Flask"


@app.route("/bargraph")
def barGraph():  # 받아오려면 매개변수 필요하겠지

    title = "bargraph"
    labels = []
    data = []

    # fetchall()로 넘어올 것
    tupledata = (
        ("A회사", "100"),
        ("B회사", "500"),
        ("C회사", "600"),
        ("D회사", "1000"),
        ("E회사", "200"),
        ("F회사", "10"),
    )

    # labels 뽑아오기
    for i in range(len(tupledata)):
        labels.append(tupledata[i][0])

    print(labels)

    # data 뽑아오기
    for i in range(len(tupledata)):
        data.append(tupledata[i][1])

    print(data)

    return render_template("bargraph.html", title=title, labels=labels, data=data)


@app.route("/linegraph")
def lineGraph():
    title = "linegraph"
    data = []

    # DB에서 받아올 월별 거래량 데이터
    monthlydata = (
        "200",
        "300",
        "600",
        "100",
        "500",
        "100",
        "150",
        "750",
        "412",
        "861",
        "40",
        "577",
    )

    for i in range(len(monthlydata)):
        data.append(monthlydata[i])
    print(data)

    return render_template("linegraph.html", title=title, data=data)


if __name__ == "__main__":
    app.run(debug=True)

from app import app
