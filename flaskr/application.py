from flask import Flask, json, render_template, redirect, url_for, request, jsonify


# EB looks for an 'application' callable by default.
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("base.html")


# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    app.debug = True
    app.run(host="0.0.0.0", port=5000)
