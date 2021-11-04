#!/bin/bash
#
# Copyright 2016-2017 ZTE Corporation.
# Copyright 2021 Orange.
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

cd /home/uuihome/uui/bert-master/ || exit

mkdir -p upload
nohup sh -c 'python -u api_squad_online.py 33011 2>&1 | tee online.log' | tee nohup.out &
nohup sh -c 'python -u api_squad_offline.py 33012 2>&1 | tee offline.log' | tee nohup.out &
nohup sh -c 'python -u upload.py 33013 2>&1 | tee upload.log' | tee nohup.out &

/usr/bin/tf_serving_entrypoint.sh
