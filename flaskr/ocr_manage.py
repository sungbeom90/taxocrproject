

# def image_upload():
#     # 사용자가 이미지를 업로드 한뒤, 해당 내용을 확인할 수 있는 함수
#     # 업로드 된 파일은 제목이 DB에 저장되어야 하며, (불러오기 위함)
#         # 파일을 저장시에는 중복방지를 위해 파일 제목 + 저장시간이 더해질 예정
#         # 제목 저장이유 : 이후에 보다 쉽게 출력하기 위함
#     # 이미지 파일 자체는 host PC에 저장할 예정입니다.
#     return None

# def decoded_image():
#     # 업로드된 이미지를 디코딩하여 바운딩박스가 그려진 이미지를 출력해줍니다.
#     # 이미지의 크기는 사용자가 업로드한 이미지와 동일한 크기로 업로드 되어야 합니다.
#     return None

# def supply_desc():
#     # 새로운 상세페이지 열어달라고 시작
#     return None

# def supply_update():
#     #이렇게 업데이트 해달라고 요청 발송??
#     return None

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