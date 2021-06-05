from flask import Flask, json, render_template, redirect, url_for, request, jsonify
import os
from werkzeug.utils import secure_filename
import mod_dbconn

def supply_desc(p_id):
    #상세페이지 조회하는 함수
    db_class = mod_dbconn.Database()
    sql = """SELECT *
                FROM taxocr.t_provider
                WHERE p_id = %s"""
    desc_dict = db_class.executeAll(sql, args=p_id)
    return desc_dict



def supply_update(args_dict, db_class):
    #공급자 수정하기
    sql = """UPDATE taxocr.t_provider
                 SET p_corp_num = %(p_corp_num)s,
                 p_corp_name = %(p_corp_name)s, p_ceo_name = %(p_ceo_name)s, p_add = %(p_add)s, p_stat=%(p_stat)s,
                 p_type = %(p_type)s, p_email = %(p_email)s
                 WHERE p_id = %(p_id)s """
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