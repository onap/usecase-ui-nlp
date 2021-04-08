#!/bin/bash
#
# Copyright 2016-2017 ZTE Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cd /home/run/
wget https://github.com/google-research/bert/archive/master.zip
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

unzip master.zip
unzip uncased_L-12_H-768_A-12.zip
rm master.zip uncased_L-12_H-768_A-12.zip
cp scripts/* bert-master/
cd /home/run/bert-master/

nohup python -u api_squad_online.py 33011 > online.log 2>&1 &
nohup python -u api_squad_offline.py 33012 > offline.log 2>&1 &

tail -f offline.log
