from flask import Flask, json, render_template, redirect, url_for, request, jsonify
import os
from werkzeug.utils import secure_filename
from app import mod_dbconn

db_class = mod_dbconn.Database()


def supply_desc(p_id):
    # 상세페이지 조회하는 함수
    db_class = mod_dbconn.Database()
    sql = """SELECT *
                FROM taxocr.t_provider
                WHERE p_id = %s"""
    desc_dict = db_class.executeAll(sql, args=p_id)
    return desc_dict


def supply_update(args_dict, db_class):
    # 공급자 수정하기
    sql = """UPDATE taxocr.t_provider
                 SET p_corp_num = %(p_corp_num)s,
                 p_corp_name = %(p_corp_name)s, p_ceo_name = %(p_ceo_name)s, p_add = %(p_add)s, p_stat=%(p_stat)s,
                 p_type = %(p_type)s, p_email = %(p_email)s
                 WHERE p_id = %(p_id)s """
    db_class.execute(query=sql, args=args_dict)
    db_class.commit()
    return args_dict


def supply_delete(p_id, db_class):
    # 공급자 삭제하기하기
    sql = """DELETE FROM taxocr.t_provider
                 WHERE p_id = %(p_id)s """
    args_dict = {"p_id": p_id}
    db_class.execute(query=sql, args=args_dict)
    db_class.commit()
    return args_dict


# def supply_update_sql(request):
#     #전달받은걸 DB에 이렇게 업데이트 하자고 DB에 전달
#         args_dict = request.form.to_dict()
#         print(args_dict)
#         sql = """INSERT into taxocr.t_bill (b_id, b_date, b_mr, b_etc, b_cost_total, b_cost_sup, b_cost_tax,
#                                             b_cost_cash, b_cost_check, b_cost_note, b_cost_credit, FK_p_id)
#                 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
#         args = (
#             args_dict["b_id"],
#             args_dict["b_date"],
#             args_dict["b_mr"],
#             args_dict["b_etc"],
#             int(args_dict["b_cost_total"].replace(",", "")),
#             int(args_dict["b_cost_sup"].replace(",", "")),
#             int(args_dict["b_cost_tax"].replace(",", "")),
#             int(args_dict["b_cost_cash"].replace(",", "")),
#             int(args_dict["b_cost_check"].replace(",", "")),
#             int(args_dict["b_cost_note"].replace(",", "")),
#             int(args_dict["b_cost_credit"].replace(",", "")),
#             args_dict["FK_p_id"],
#         )
#         print(args)
#         db_class.execute(query=sql, args=args)
#         db_class.commit()

#     return None


# 파일 제목을 DB에 insert 하는 함수
# def file_info_save():
#     sql = "insert * into [table] from locations"
#     cursor = conn.cursor()
#     cursor.execute(sql)
#     = cursor.fetchall()
#     return geoinfo

# def file_info_read():
#     sql = "select * from files"
#     cursor = conn.cursor()
#     cursor.execute(sql)
#     file_list = cursor.fetchall()
#     return file_list


# ==================== 박성범 작성 ===========================

# 공급자 존재여부 조회하는 함수
def provider_exists(p_id):
    print("{} 가 db에 있는지 확인중.. ".format(p_id["p_id"]))
    sql = """SELECT EXISTS (select * from t_provider where p_id=%(p_id)s) as success"""
    try:
        desc_dict = db_class.executeOne(sql, args=p_id)
        print("{} 가 db에 있는지 결과 : {} ".format(p_id["p_id"], desc_dict["success"]))
        return desc_dict["success"]  # provider_exists 요청 성공
    except:
        print("provider_exists 요청 실패")
        return -1  # provider_exists 요청 실패


# 공급자 입력(생성)하는 함수
def provider_insert(t_provider):
    print("공급자 등록 요청 접수됨")
    print(t_provider)
    sql = """INSERT into taxocr.t_provider (p_id, p_corp_num, p_corp_name, p_ceo_name, p_add, p_stat, p_type, p_email)
            VALUES (%(p_id)s,%(p_corp_num)s,%(p_corp_name)s,%(p_ceo_name)s,%(p_add)s,%(p_stat)s,%(p_type)s,%(p_email)s)"""
    try:
        db_class.execute(query=sql, args=t_provider)
        db_class.commit()
        print("{} 공급자 db 등록 완료".format(t_provider["p_id"]))
        return 1  #  provider_insert 요청 성공
    except:
        print("provider_insert 요청 실패")
        return -1  # provider_insert 요청 실패


# 계산서 입력(생성)하는 함수
def bill_insert(t_bill):
    print("계산서 등록 요청 접수됨")
    print(t_bill)
    sql = """INSERT into taxocr.t_bill (b_id, b_date, b_mr, b_etc, b_cost_total, b_cost_sup, b_cost_tax,
                                            b_cost_cash, b_cost_check, b_cost_note, b_cost_credit, FK_p_id)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    try:
        db_class.execute(query=sql, args=t_bill)
        db_class.commit()
        print("{} 계산서 db 등록 완료".format(t_bill[0]))
        return 1  # bill_insert 요청 성공
    except:
        print("bill_insert 요청 실패")
        return -1  # bill_insert 요청 실패


# 계산서 가격 ',' 제거 함수
def cost_replace(cost_str):
    result = 0
    if cost_str is "":
        return result
    else:
        result = int(cost_str.replace(",", ""))
        return result


# =================== 최지영 작성 =================
year_ = "2010"  # 이거 웹에서 받아올거라 변수 여기서 선언 안되어있어도 됨(?)

# 세금 데이터----------------------------------------
def taxdata(year_):
    taxsql = """SELECT SUM(t_bill.b_cost_tax)
                FROM t_bill
                WHERE YEAR(t_bill.b_date) = '2010' """

    taxrow = db_class.executeOne(taxsql, year_)
    taxdata = int(taxrow["SUM(t_bill.b_cost_tax)"])

    return taxdata


# Doughnut graph------------------------------------
def doughnutGraph(year_):
    labels = []  # 그래프 x축: 회사명
    data = []  # 그래프 y축 : 거래 금액
    temp = []  # 디비에서 빼온 것 잠깐 담을 리스트
    temp2 = []  # 위와 동일
    temptuple = []  # 위와 동일

    sql = """SELECT t_provider.p_corp_name as p_corp_name,
            SUM(t_bill.b_cost_total) as b_cost_total_sum
            FROM t_provider, t_bill
            WHERE t_provider.p_id = t_bill.FK_p_id AND YEAR(t_bill.b_date) = '2010'
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
    data2 = []  # 그래프 y축 : 거래 금액 (x축은 javascript로 고정)
    linesql = """SELECT MONTH(t_bill.b_date), SUM(t_bill.b_cost_total)
                FROM t_bill
                WHERE YEAR(t_bill.b_date) = '2010'
                GROUP BY MONTH(t_bill.b_date)"""
    linerow = db_class.executeAll(linesql, year_)

    print("선그래프 데이터 : ", linerow)

    # dict에서 금액 정보만 빼오기
    for i in linerow:
        data2.append(int(i["SUM(t_bill.b_cost_total)"]))

    return data2


# Bar graph-----------------------------------------------
def barGraph(year_):
    data3 = []  # 그래프 y축 : 거래 금액 (x축은 javascript로 고정)
    barsql = """SELECT SUM(t_bill.b_cost_total),
                SUM(t_bill.b_cost_cash),
                SUM(t_bill.b_cost_check),
                SUM(t_bill.b_cost_note),
                SUM(t_bill.b_cost_credit) 
        FROM t_bill
        WHERE YEAR(t_bill.b_date) = '2010'  """

    barrow = db_class.executeAll(
        barsql, year_
    )  # [{'b_cost_total':50000,'b_cost_cash':50000,'b_cost_check':50000,'b_cost_note':50000, 'b_cost_credit':50000}, {}, ...]

    print("막대그래프 데이터 : ", barrow)

    for i in barrow:
        data3.append(int(i["SUM(t_bill.b_cost_cash)"]))
        data3.append(int(i["SUM(t_bill.b_cost_check)"]))
        data3.append(int(i["SUM(t_bill.b_cost_note)"]))
        data3.append(int(i["SUM(t_bill.b_cost_credit)"]))

    return data3
