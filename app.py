from flask import Flask , render_template , session , request , redirect, url_for
import os
from model import result

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/CustSegPage')
def search():
    return render_template('CustSegPage.html')

@app.route('/output', methods = ['GET', 'POST'])
def output():
    if request.method == 'POST':
        income = request.form['income']
        spending = request.form['spending']
        ClusterNumber = result(income,spending)
        return render_template("output.html", ClusterNumber=ClusterNumber)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5005)
