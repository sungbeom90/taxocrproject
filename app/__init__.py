from flask import Flask, render_template
from app import mod_dbconn

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/db")
def select():
    db_class = mod_dbconn.Database()
    sql = "SELECT b_id \
                FROM taxocr.t_provider"
    row = db_class.executeAll(sql)
    print(row)
    return render_template("db.html", resultData=row[0])


if __name__ == "__main__":
    app.run(debug=True)

from app import app
