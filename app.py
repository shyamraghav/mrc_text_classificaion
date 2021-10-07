# Importing the dependant libraries

from flask import Flask, request
from classifier import MrCooper
import json

app = Flask(__name__)

cooper = MrCooper()


@app.route('/')
def health():
    return "OK"


@app.route('/classify/', methods=['GET', 'POST'])
def respond():
    if request.method == 'GET':
        return "This request is made for POST CALL. Please call as POST with payload {text: actual_text}"
    if request.method == "POST":
        input_text = json.loads(request.data)
        print(input_text['text'])
        output = cooper.classify(str(input_text['text']))
        # print("output :",output)
        return output


if __name__ == '__main__':
    print("Initializing Flask Application")
    app.run(port='5555')
