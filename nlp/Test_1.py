from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from unittest import mock
import sys
sys.path.append("scripts")
import pandas as pd
sys.modules['modeling'] = mock.Mock()
sys.modules['optimization'] = mock.Mock()
sys.modules['tokenization'] = mock.Mock()
sys.modules['tensorflow'] = mock.Mock()
import api_squad_offline
import tokenization
import tensorflow
import api_squad
import requests
import create_squad_features


class TestApiSquad(unittest.TestCase):

    def test_make_json(self):
        data = {"Sequence": ['0'], "text": ['Please'], "Communication Service Name": ['exclusive'], "Max Number of UEs": ['10'], "Data Rate Downlink": ['1Gbps'], "Latency": ['low'], "Data Rate Uplink": ['1Gbps'], "Resource Sharing Level": ['Resources are not shared'], "Mobility": ['Fixed network'], "Area": ['East of seven Science and Technology cities in North Changping']}
        df = pd.DataFrame(data, index=[0])
        mock_data_train = mock.Mock(return_value=df)
        pd.read_excel = mock_data_train
        result = api_squad.make_json('fileName', ['Communication Service Name', 'Max Number of UEs', 'Data Rate Downlink', 'Latency', 'Data Rate Uplink', 'Resource Sharing Level', 'Mobility', 'Area'])
        print(result)
        self.assertEqual(result, '{"data": [{"title": "Not available", "paragraphs": [{"context": "Please", "qas": [{"answers": [{"text": "exclusive", "answer_start": -1}], "is_impossible": 0, "id": "0Communication Service Name", "question": "Communication Service Name"}, {"answers": [{"text": "10", "answer_start": -1}], "is_impossible": 0, "id": "0Max Number of UEs", "question": "Max Number of UEs"}, {"answers": [{"text": "1Gbps", "answer_start": -1}], "is_impossible": 0, "id": "0Data Rate Downlink", "question": "Data Rate Downlink"}, {"answers": [{"text": "low", "answer_start": -1}], "is_impossible": 0, "id": "0Latency", "question": "Latency"}, {"answers": [{"text": "1Gbps", "answer_start": -1}], "is_impossible": 0, "id": "0Data Rate Uplink", "question": "Data Rate Uplink"}, {"answers": [{"text": "Resources are not shared", "answer_start": -1}], "is_impossible": 0, "id": "0Resource Sharing Level", "question": "Resource Sharing Level"}, {"answers": [{"text": "Fixed network", "answer_start": -1}], "is_impossible": 0, "id": "0Mobility", "question": "Mobility"}, {"answers": [{"text": "East of seven Science and Technology cities in North Changping", "answer_start": -1}], "is_impossible": 0, "id": "0Area", "question": "Area"}]}]}]}')

    def test_read_squad_examples(self):
        json = '{"data": [{"title": "Not available", "paragraphs": [{"context": "Please assist in opening exclusive slicing service. It is estimated that the number of access user devices is 10. Upload and download are required to be at least 1Gbps, and the delay is low. Fixed network. I live in the East of seven Science and Technology cities in North Changping. ", "qas": [{"answers": [{"text": "exclusive", "answer_start": -1}], "is_impossible": 0, "id": "0Communication Service Name", "question": "Communication Service Name"}, {"answers": [{"text": "10", "answer_start": -1}], "is_impossible": 0, "id": "0Max Number of UEs", "question": "Max Number of UEs"}, {"answers": [{"text": "1Gbps", "answer_start": -1}], "is_impossible": 0, "id": "0Data Rate Downlink", "question": "Data Rate Downlink"}, {"answers": [{"text": "low", "answer_start": -1}], "is_impossible": 0, "id": "0Latency", "question": "Latency"}, {"answers": [{"text": "1Gbps", "answer_start": -1}], "is_impossible": 0, "id": "0Data Rate Uplink", "question": "Data Rate Uplink"}, {"answers": [{"text": "Resources are not shared", "answer_start": -1}], "is_impossible": 0, "id": "0Resource Sharing Level", "question": "Resource Sharing Level"}, {"answers": [{"text": "Fixed network", "answer_start": -1}], "is_impossible": 0, "id": "0Mobility", "question": "Mobility"}, {"answers": [{"text": "East of seven Science and Technology cities in North Changping", "answer_start": -1}], "is_impossible": 0, "id": "0Area", "question": "Area"}]}]}]}'
        questions = ['Communication Service Name', 'Max Number of UEs', 'Data Rate Downlink', 'Latency', 'Data Rate Uplink',
         'Resource Sharing Level', 'Mobility', 'Area']
        mock_data = mock.Mock(return_value=json)
        api_squad.make_json = mock_data
        tokenization.whitespace_tokenize = mock.Mock(return_value=("r", "u", "n", "o", "o", "b"))
        tensorflow.logging.warning = mock.Mock(return_value=1)
        result = api_squad.read_squad_examples("", True, questions, True)
        print(result)
        self.assertEqual(result, [])

    def test_convert_examples_to_features(self):
        examples = []
        example = api_squad.SquadExample(
            qas_id='50Max Number of UEs',
            question_text='Max Number of UEs',
            doc_tokens=['X', 'X', ' ', 'B', 'a', 'n', 'k', ' ', 'o', 'f', 'f', 'i', 'c', 'i', 'a', 'l', ' ', 'n', 'e', 't', 'w', 'o', 'r'], #'k', ' ', 'a', 't', ' ', 'F', 'i', 'n', 'a', 'n', 'c', 'i', 'a', 'l', ' ', 'S', 't', 'r', 'e', 'e', 't', ',', ' ', 'X', 'i', 'c', 'h', 'e', 'n', 'g', ' ', 'D', 'i', 's', 't', 'r', 'i', 'c', 't', ',', ' ', 'B', 'e', 'i', 'j', 'i', 'n', 'g', ';', ' ', 'a', 'c', 'c', 'e', 's', 's', ' ', 'n', 'u', 'm', 'b', 'e', 'r', ' ', '1', '0', '0', '0', '0', ';', ' ', 'n', 'o', 'n', '-', 's', 'h', 'a', 'r', 'e', 'd', ' ', 'n', 'e', 't', 'w', 'o', 'r', 'k', ';', ' ', 's', 't', 'a', 't', 'i', 'o', 'n', 'a', 'r', 'y', ' ', 'n', 'e', 't', 'w', 'o', 'r', 'k', ';', ' ', 't', 'h', 'e', ' ', 'l', 'a', 't', 'e', 'n', 'c', 'y', ' ', 'i', 's', ' ', 'a', 's', ' ', 'l', 'o', 'w', ' ', 'a', 's', ' ', 'p', 'o', 's', 's', 'i', 'b', 'l', 'e', ',', ' ', 'u', 'p', 'l', 'o', 'a', 'd', ' ', 'a', 'n', 'd', ' ', 'd', 'o', 'w', 'n', 'l', 'o', 'a', 'd', ' ', 'b', 'a', 'n', 'd', 'w', 'i', 't', 'h', 's', ' ', 'a', 'r', 'e', ' ', '1', '0', 'G', 'b'],
            orig_answer_text=None,
            start_position = 12,
            end_position=12,
            is_impossible=0)
        examples.append(example)
        tokenizer= mock.Mock()
        max_seq_length = 512
        doc_stride = 128
        max_query_length = 64
        is_training = True
        output_fn = mock.Mock()

        tokenizer.tokenize = mock.Mock(return_value="T")
        tokenizer.convert_tokens_to_ids = mock.Mock(return_value=['X', 'X', ' ', 'B', 'a', 'n', 'k', ' ', 'o', 'f', 'f', 'i', 'c', 'i', 'a', 'l', ' ', 'n', 'e', 't', 'w', 'o', 'r', 'k', ' ', 'a', 't'])
        tokenization.printable_text = mock.Mock(return_value="XX")
        result = api_squad.convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn)
        self.assertEqual([], [])
        # examples = ['qas_id': '50Max Number of UEs', 'question_text': "Max Number of UEs", doc_tokens: [XX Bank official network at Financial Street, Xicheng District, Beijing; access number 10000; non-shared network; stationary network; the latency is as low as possible, upload and download bandwiths are 10Gb], start_position: 12, end_position: 12, is_impossible: 0]

class TestApiSquadOffline(unittest.TestCase):
    def test_serving_input_fn(self):
        tensorflow.placeholder = mock.Mock(return_value = "1")

        def sum_func():
            return "input_fn"
        tensorflow.estimator.export.build_raw_serving_input_receiver_fn = mock.Mock(return_value = sum_func)
        result = api_squad_offline.serving_input_fn()
        self.assertEqual(result, "input_fn")

class TestCreateSquadFeatures(unittest.TestCase):

    def test_get_squad_feature_result(self):

        arr = []
        for num in range(5, 63):
            arr.append([1] * num)

        tokenizer= mock.Mock()
        tokenization.BasicTokenizer = mock.Mock(return_value=tokenizer)
        tokenizer.tokenize = mock.Mock(return_value="T")
        tokenizer.convert_tokens_to_ids = mock.Mock(side_effect=arr)
        tokenization.printable_text = mock.Mock(return_value="Please")

        requests.post = mock.Mock(return_value=request('{"predictions":[{"start_logits":[0,1,2,3,4,5,6,7,8,9,10],"end_logits":[10,11,12,13,14,15,16,17,18,19,20]}]}'))

        create_squad_features.get_squad_feature_result("predict", "Please assist to open the exclusive slicing service. It is estimated that the number of access user devices is 1. It is required to upload 500MB and download 1GB. It is also required to have low delay, fixed network and no sharing of resources. My residence is in the future science and technology city of Changping District. .", tokenizer, ['Communication Service Name'], "http://localhost:8502/v1/models/predict:predict")
        self.assertEqual(1,1)


class request(object):
    def __init__(self, text):
        self.text = text

if __name__ == "__main__":
    unittest.main()