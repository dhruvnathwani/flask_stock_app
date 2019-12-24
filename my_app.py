from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import json
from stocks import get_current_price, predict_next_price


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'index'

@app.route('/first')
def first():
    return 'hello'

@app.route('/second/<first>/<second>/', methods = ['GET'])
def second(first,second):

    output = int(first) + int(second)

    output = {'output': str(output)}

    return jsonify(output)

@app.route('/stocks/<stock>/',methods = ['GET'])
def stocks(stock):
    #price = get_current_price(stock)

    #output = {'current_price': str(price)}

    current_price, predicted_price, method_used = predict_next_price(stock)

    #info = info[0]

    output = {'current_price': str(current_price[0][0]),
    'predicted_price': str(predicted_price[0]),
    'method_used':str(method_used)}

    return jsonify(output) 

    #return info




if __name__ == '__main__':
    app.run(debug=True)