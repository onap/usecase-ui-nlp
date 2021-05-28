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
import api_squad


class TestApiSquad(unittest.TestCase):

    def test_make_json(self):
        data = {"Sequence": ['0'], "text": ['Please'], "Communication Service Name": ['exclusive'], "Max Number of UEs": ['10'], "Data Rate Downlink": ['1Gbps'], "Latency": ['low'], "Data Rate Uplink": ['1Gbps'], "Resource Sharing Level": ['Resources are not shared'], "Mobility": ['Fixed network'], "Area": ['East of seven Science and Technology cities in North Changping']}
        df = pd.DataFrame(data, index=[0])
        mock_data_train = mock.Mock(return_value=df)
        pd.read_excel = mock_data_train
        result = api_squad.make_json('fileName', ['Communication Service Name', 'Max Number of UEs', 'Data Rate Downlink', 'Latency', 'Data Rate Uplink', 'Resource Sharing Level', 'Mobility', 'Area'])
        print(result)
        self.assertEqual(result, '{"data": [{"title": "Not available", "paragraphs": [{"context": "Please", "qas": [{"answers": [{"text": "exclusive", "answer_start": -1}], "is_impossible": 0, "id": "0Communication Service Name", "question": "Communication Service Name"}, {"answers": [{"text": "10", "answer_start": -1}], "is_impossible": 0, "id": "0Max Number of UEs", "question": "Max Number of UEs"}, {"answers": [{"text": "1Gbps", "answer_start": -1}], "is_impossible": 0, "id": "0Data Rate Downlink", "question": "Data Rate Downlink"}, {"answers": [{"text": "low", "answer_start": -1}], "is_impossible": 0, "id": "0Latency", "question": "Latency"}, {"answers": [{"text": "1Gbps", "answer_start": -1}], "is_impossible": 0, "id": "0Data Rate Uplink", "question": "Data Rate Uplink"}, {"answers": [{"text": "Resources are not shared", "answer_start": -1}], "is_impossible": 0, "id": "0Resource Sharing Level", "question": "Resource Sharing Level"}, {"answers": [{"text": "Fixed network", "answer_start": -1}], "is_impossible": 0, "id": "0Mobility", "question": "Mobility"}, {"answers": [{"text": "East of seven Science and Technology cities in North Changping", "answer_start": -1}], "is_impossible": 0, "id": "0Area", "question": "Area"}]}]}]}')


if __name__ == "__main__":
    unittest.main()