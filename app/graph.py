from app import mod_dbconn

year_='2010' # 이거 웹에서 받아올거라 변수 여기서 선언 안되어있어도 됨(?)
db_class = mod_dbconn.Database()

# 세금 데이터----------------------------------------
def taxdata(year_):
    taxsql = """SELECT SUM(t_bill.b_cost_tax)
                FROM t_bill
                WHERE YEAR(t_bill.b_date) = %s""" 

    taxrow = db_class.executeOne(taxsql, year_)
    taxdata = int(taxrow["SUM(t_bill.b_cost_tax)"])

    return taxdata

# Doughnut graph------------------------------------
def doughnutGraph(year_):
    labels = [] # 그래프 x축: 회사명
    data = [] #그래프 y축 : 거래 금액
    temp = [] # 디비에서 빼온 것 잠깐 담을 리스트
    temp2 = [] # 위와 동일
    temptuple = [] # 위와 동일

    sql = """SELECT t_provider.p_corp_name as p_corp_name,
            SUM(t_bill.b_cost_total) as b_cost_total_sum
            FROM t_provider, t_bill
            WHERE t_provider.p_id = t_bill.FK_p_id AND YEAR(t_bill.b_date) = %s
            GROUP BY FK_p_id
            ORDER BY SUM(t_bill.b_cost_total) DESC"""
    row = db_class.executeAll(sql, year_)

    print("도넛그래프 데이터 : ", row)

    for i in row:
        temp.append(i["p_corp_name"])
        temp2.append(int(i["b_cost_total_sum"]))

    # 회사명, 거래금액
    for i in range(len(temp)):
        a = (temp[i], temp2[i])
        temptuple.append(a)

    # 회사명 뽑아오기
    for i in range(len(temptuple)):
        labels.append(temptuple[i][0])

    # 거래 금액
    for i in range(len(temptuple)):
        data.append(temptuple[i][1])

    return labels, data
# Line graph---------------------------------------------
def lineGraph(year_):
    data2 = [] # 그래프 y축 : 거래 금액 (x축은 javascript로 고정)
    linesql = """SELECT MONTH(t_bill.b_date), SUM(t_bill.b_cost_total)
                FROM t_bill
                WHERE YEAR(t_bill.b_date) = %s 
                GROUP BY MONTH(t_bill.b_date)"""
    linerow = db_class.executeAll(linesql, year_) 

    print("선그래프 데이터 : ", linerow)
    
    # dict에서 금액 정보만 빼오기
    for i in linerow:
        data2.append(int(i["SUM(t_bill.b_cost_total)"]))

    return data2
# Bar graph-----------------------------------------------
def barGraph(year_): 
    data3 = [] # 그래프 y축 : 거래 금액 (x축은 javascript로 고정)
    barsql = """SELECT SUM(t_bill.b_cost_total),
                SUM(t_bill.b_cost_cash),
                SUM(t_bill.b_cost_check),
                SUM(t_bill.b_cost_note),
                SUM(t_bill.b_cost_credit) 
        FROM t_bill
        WHERE YEAR(t_bill.b_date) = %s'""" 

    barrow = db_class.executeAll(barsql, year_) #[{'b_cost_total':50000,'b_cost_cash':50000,'b_cost_check':50000,'b_cost_note':50000, 'b_cost_credit':50000}, {}, ...]

    print("막대그래프 데이터 : ",barrow)

    for i in barrow:
        data3.append(int(i["SUM(t_bill.b_cost_cash)"]))
        data3.append(int(i["SUM(t_bill.b_cost_check)"]))  
        data3.append(int(i["SUM(t_bill.b_cost_note)"]))  
        data3.append(int(i["SUM(t_bill.b_cost_credit)"]))
    
    return data3


#-------------이 이하부터는 __init__.py에서 대체되어야 할 부분입니다.
from app import graph

@app.route("/bargraph", methods=["POST"])
def berGraph(): # 매개변수 받아야하나? ㄴㄴ
    year_ = request.form.values() # year_ 값
    print(year_)

    # 세금 데이터
    taxdata = graph.taxdata(year_)

    # 도넛 그래프
    labels, data = graph.doughnutGraph(year_)

    # 라인 그래프
    data2 = graph.lineGraph(year_)

    #막대 그래프
    data3 = graph.barGraph(year_)

    return render_template(
        "bargraph.html",
        title=title,
        taxdata=taxdata,
        labels=labels,
        data=data,
        data2=data2,
        data3=data3,
    )
