# import cx_Oracle as oci

# oracle_dsn = oci.makedsn(host="192.168.2.247", port=1521, sid="orcl")
# conn = oci.connect(dsn=oracle_dsn, user="emg", password="1234566")

def image_upload():
    # 사용자가 이미지를 업로드 한뒤, 해당 내용을 확인할 수 있는 함수
    # 업로드 된 파일은 제목이 DB에 저장되어야 하며, (불러오기 위함)
        # 파일을 저장시에는 중복방지를 위해 파일 제목 + 저장시간이 더해질 예정
        # 제목 저장이유 : 이후에 보다 쉽게 출력하기 위함
    # 이미지 파일 자체는 host PC에 저장할 예정입니다.
    return None

def decoded_image():
    # 업로드된 이미지를 디코딩하여 바운딩박스가 그려진 이미지를 출력해줍니다.
    # 이미지의 크기는 사용자가 업로드한 이미지와 동일한 크기로 업로드 되어야 합니다.
    return None


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