path=$1
use_gpu=$2
export CUDA_VISIBLE_DEVICES=$use_gpu
netstat -nap | grep 8502 | awk 'NR==1{printf $7}' | sed 's/\([0-9]*\).*/\1/g' | xargs kill -9
sleep 5
nohup tensorflow_model_server --port=8500 --rest_api_port=8502 --model_name=predict --model_base_path=$path > server.log 2>&1 &