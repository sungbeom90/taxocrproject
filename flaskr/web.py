from flask import Flask, json, render_template, redirect, url_for, request, jsonify
import ocr_manage as om
import os
from werkzeug.utils import secure_filename
import mod_dbconn

app = Flask(__name__)

# 업로드된 파일주소가 저장되는 리스트
upload_file_list = []
app.config['UPLOAD_DIR'] = "./static/image/"

@app.route("/")
def home():
    return render_template("home.html")

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
    #image_path2 = "./image/after_0.jpg"
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

@app.route("/supply", methods=["GET"])
def supply():
    return render_template("supply.html")


@app.route("/predicted_img", methods=["POST"])
def Predict_img():
    # 예측된 이미지에서 text를 출력해서 보내왔습니다.
    jsonData = request.get_json()
    data1 = str(jsonData["testkey"])
    return jsonData

@app.route('/con_base', methods=['GET'])
def con_base():
    #업로드된 사진 출력 
    image_upload = upload_file_list.pop()
    return render_template("con_base.html", image_file=image_upload)

@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method =='GET':
        return render_template('home.html')
    elif request.method =='POST':
        Data = request.get_json()
        print(Data)
        Data1 = str(Data['test1'])
        print(Data1)
        return render_template('test1.html', Data1=Data1)

@app.route("/supply_db")
def select_sup():
    db_class = mod_dbconn.Database()
    sql = "SELECT *\
                FROM taxocr.t_provider"
    all_sup_dict = db_class.executeAll(sql)
    print(all_sup_dict)
    dataNum = len(all_sup_dict)
    return render_template("supply.html", resultData=all_sup_dict, dataNum=dataNum)

@app.route("/supply_desc", methods=['GET'])
def select_sup_desc():
    p_id = request.args.get('p_id')
    desc_dict = om.supply_desc(p_id)
    return render_template("supply_desc.html", desc_dict=desc_dict[0])

@app.route("/supply_desc_update")
# 아직 미완임 sql 작성관련으로 보여 일단 stop
def update_sup():
   om.supply_update_sql(request)
   return render_template("")

@app.route("/update_provider", methods=("GET", "POST"))
def update_provider():
    db_class = mod_dbconn.Database()
    
    print("수정요청 접수됨")
    if request.method == "POST":
        args_dict = request.form.to_dict()
        args = tuple(request.form.values())
        print(args_dict)
        sql = """UPDATE taxocr.t_provider
                 SET p_id = %s, p_corp_num = %s,
                 p_corp_name = %s, p_ceo_name = %s, p_add = %s, p_stat=%s,
                 p_type = %s, p_email = %s
                 WHERE p_id = %s """
        db_class.execute(query=sql, args=args)
        db_class.commit()
        return render_template("supply.html")


if __name__ == "__main__":
    app.debug = True
    #app.run(port=80)
    app.run(host="192.168.187.1", port=80, debug=True)
