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
        db_class = mod_dbconn.Database()
        args_dict = request.form.to_dict()
        print(args)
        sql = """INSERT into taxocr.t_bill (b_id, b_date, b_mr, b_etc, b_cost_total, b_cost_sup, b_cost_tax, *\
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
        db_class.execute(query=sql, args=args)
        db_class.commit()
        return render_template("home.html")


@app.route("/")
def printHello():
    return "Hello World - Flask"


@app.route("/bargraph")
def barGraph():  # 받아오려면 매개변수 필요하겠지
    title = "bargraph"
    labels = [] # 도넛그래프 x축 : 회사명
    data = []
    data2 = []
    # 미수금 데이터
    misoo = 60000000

    # db 테스트
    # Doughnut graph
    temp = []
    temptuple=[]
    db_class = mod_dbconn.Database()
    sql = "SELECT p_corp_name\
                FROM taxocr.t_provider"
    row = db_class.executeAll(sql)
    print(row)  # [{'p_corp_name':'주식회사 아이피스'},{'p_corp_name':'(주)타라그래픽스 동여의도점'}, ...]

    for i in row:
        temp.append(i['p_corp_name'])
    print(temp)

    # 회사명, 거래금액         
    for i in range(len(temp)):
        a = (temp[i], '200')
        temptuple.append(a)    
    print(temptuple)
    
    # 회사명 뽑아오기
    for i in range(len(temptuple)):
        labels.append(temptuple[i][0])

    print(labels)

    #거래 금액
    for i in range(len(temptuple)):
        data.append(temptuple[i][1])

    print(data)

    
    # fetchall()로 넘어올 것
    tupledata = (
        ("A회사", "300"),
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
    # Line graph
    # data 뽑아오기
    for i in range(len(tupledata)):
        data.append(tupledata[i][1])
    print(data)
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
        data2.append(monthlydata[i])
    print(data2)

    # Bubble graph
    return render_template(
        "bargraph.html",
        title=title,
        misoo=misoo,
        labels=labels,
        data=data,
        data2=data2
    )

if __name__ == "__main__":
    app.run(debug=True)
from app import app
