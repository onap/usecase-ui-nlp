#!/usr/bin/env python
# coding: utf-8

# auther = 'liuzhiyong'
# date = 20201204


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import sys
from flask import Flask, abort, request, jsonify

import os
from global_setting import tokenizer_ch, CUDA_VISIBLE_DEVICES
from create_squad_features import get_squad_feature_result
from transformers import pipeline


app = Flask(__name__)


class AI2Flask:

    def __init__(self, port=5000, workers=4):
        self.app = app
        self.port = port

        @app.route('/api/online/predict', methods=['POST'])
        def text_analyse():
            if not request.json:
                abort(400)

            else:
                try:
                    try:
                        title = request.json['title']
                    except:
                        title = 'Not available'
                    text_origin = request.json['text']
                    questions = request.json['questions']

                    if len(text_origin) > 800:
                        text = text_origin[:800]
                    else:
                        text = text_origin

                    result = {}
                    for ques in questions:
                        tmp = get_squad_feature_result(title=title, text=text, tokenizer=tokenizer_ch, question=[ques], url='http://localhost:8502/v1/models/predict:predict')
                        result[ques] = dict(tmp)[ques]

                    print('finished!!')
                    return json.dumps(result)

                except KeyError as e:
                    return jsonify({"Des": 'KeyError: {}'.format(str(e)), "Result": 'None', "Status": "Error"})
                except Exception as e:
                    return jsonify({"Des": str(e), "Result": 'None', "Status": "Error"})

        @app.route('/api/online/load', methods=['POST'])
        def load_model():
            if not request.json:
                abort(400)
            else:
                try:
                    path = request.json['path']
                    flag = os.system('./load_model.sh ' + path + ' ' + CUDA_VISIBLE_DEVICES)
                    if flag == 0:
                        return jsonify({"Des": "Model loaded successfully !", "Status": "OK"})
                    else:
                        return jsonify({"Des": "Model loaded failed , check the logs !", "Status": "Error"})
                except Exception as e:
                    return jsonify({"Des": str(e), "Status": "Error"})

    def start(self):
        self.app.run(host="0.0.0.0", port=self.port, threaded=True)

class QuestionAnswerer:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_file(self):
        res = []
        read_res = csv.reader(open(self.file_path, "r", encoding="UTF-8-sig"))
        for row in read_res:
            res.append(row)
        return res

    def answerer(self, res):
        question_answerer = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')
        questions = [
            "How much bandwidth is required?",
            "Where are you from?",
            "What is your destination?"
        ]
        for context in res:
            result = question_answerer(question=questions[0], context=context[0])
            result2 = question_answerer(question=questions[1], context=context[0])
            result3 = question_answerer(question=questions[2], context=context[0])
            print(
                questions[0],
                f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
            print(
                questions[1],
                f"Answer: '{result2['answer']}', score: {round(result2['score'], 4)}, start: {result2['start']}, end: {result2['end']}")
            print(
                questions[2],
                f"Answer: '{result3['answer']}', score: {round(result3['score'], 4)}, start: {result3['start']}, end: {result3['end']}")




if __name__ == '__main__':
    port = sys.argv[1]
    AI2Flask(port=port).start()
    file_path = '../TestFile.csv'
    QA = QuestionAnswerer(file_path)
    res = QA.read_file()

    QA.answerer(res)
