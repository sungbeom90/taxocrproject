from flask import Flask, json, render_template, redirect, url_for, request, jsonify
import ocr_manage as om
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 업로드된 파일주소가 저장되는 리스트
upload_file_list = []
app.config['UPLOAD_DIR'] = "./static/image/"

@app.route("/")
def home():
    return render_template("home.html")


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
        return redirect(url_for("file_print"))


@app.route("/guide", methods=["GET"])
def guide():
    return render_template("guide.html")


@app.route("/predicted_img", methods=["POST"])
def Predict_img():
    # 예측된 이미지에서 text를 출력해서 보내왔습니다.
    jsonData = request.get_json()
    data1 = str(jsonData["testkey"])
    return jsonData


if __name__ == "__main__":
    app.debug = True
    #app.run(port=80)
    app.run(host="192.168.187.1", port=80, debug=True)
