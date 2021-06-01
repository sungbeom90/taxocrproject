from flask import Flask, render_template, redirect, url_for, request, jsonify
import ocr_manage as om
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

#업로드된 파일주소가 저장되는 리스트
upload_file_list = []

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/image_request', methods=['GET'])
def image_input():
    img = om.image_upload()
    return render_template("service.html")

@app.route('/guide', methods=['GET'])
def guide():
    return render_template("guide.html")

@app.route('/done')
def file_print():
    #이미지 경로 설정
    image_path1 = upload_file_list.pop()
    #image_path2는 text파일로 변동 예정임
    image_path2 = "./image/after_0.jpg"
    return render_template("done.html", image_file=image_path1, image_file2=image_path2)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
       # 파일 저장
      f_list = request.files.getlist('cma_file[]')
      for fil in f_list :
          fil.save("./static/image/" + secure_filename(fil.filename))
          upfile_address = "image/"+ secure_filename(fil.filename)
          upload_file_list.append(upfile_address)
          print(upfile_address)
        # 업로드된 파일명
      return redirect(url_for('file_print'))
   elif request.method =='GET':
      return '404'



if __name__ == "__main__":
    app.debug = True
    #app.run(port=80) 
    app.run(host="192.168.187.1", port=80)