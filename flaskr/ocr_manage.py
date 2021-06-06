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

def supply_delete(p_id, db_class):
    #공급자 삭제하기하기
    sql = """DELETE FROM taxocr.t_provider
                 WHERE p_id = %(p_id)s """
    args_dict = {'p_id' : p_id}
    db_class.execute(query=sql, args=args_dict)
    db_class.commit()
    return args_dict


