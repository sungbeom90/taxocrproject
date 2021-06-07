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
    # sql = """INSERT into taxocr.t_bill (b_id, b_date, b_mr, b_etc, b_cost_total, b_cost_sup, b_cost_tax,
    #                                     b_cost_cash, b_cost_check, b_cost_note, b_cost_credit, FK_p_id)
    #         VALUES (%(b_id)s,%(b_date)s,%(b_mr)s,%(b_etc)s,%(b_cost_total)s,%(b_cost_sup)s,%(b_cost_tax)s,%(b_cost_cash)s,%(b_cost_note)s,%(b_cost_credit)s,%(FK_p_id)s)"""
    sql = """INSERT into taxocr.t_bill (b_id, b_date, b_mr, b_etc, b_cost_total, b_cost_sup, b_cost_tax,
                                            b_cost_cash, b_cost_check, b_cost_note, b_cost_credit, FK_p_id)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    # try:
    db_class.execute(query=sql, args=t_bill)
    db_class.commit()
    print("{} 계산서 db 등록 완료".format(t_bill[0]))
    return 1  # bill_insert 요청 성공
    # except:
    #     print("bill_insert 요청 실패")
    #     return -1  # bill_insert 요청 실패


# 계산서 가격 ',' 제거 함수
def cost_replace(cost_str):
    result = 0
    if cost_str is "":
        return result
    else:
        result = int(cost_str.replace(",", ""))
        return result
