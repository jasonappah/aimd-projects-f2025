from flask import Flask, render_template, jsonify, request
import subprocess
import random
# from care import start

app = Flask(__name__)

#----------------------------------------------------#
#flask integration
@app.route('/')
def home():
    return render_template('hypoglycemia_portal.html')


if __name__ == "__main__":
    app.run(debug=True)