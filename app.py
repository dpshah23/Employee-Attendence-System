from flask import *

app=Flask(__name__)

@app.route('/')
def main():
    return render_template('main.html')