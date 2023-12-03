path=$1
use_gpu=$2
export CUDA_VISIBLE_DEVICES=$use_gpu
netstat -nap | grep 8502 | awk 'NR==1{printf $7}' | sed 's/\([0-9]*\).*/\1/g' | xargs kill -9
sleep 5
nohup tensorflow_model_server --port=8500 --rest_api_port=8502 --model_name=predict --model_base_path=$path > server.log 2>&1 &


 #模型训练完后，往往需要将模型应用到生产环境中。最常见的就是通过TensorFlow Serving来将模型部署到服务器端，以便客户端进行请求访问。
 #tensorflow_model_server \
 #    --rest_api_port=8501 \
 #    --model_name=VGG16 \
 #    --model_base_path="/home/.../.../saved"  # 文件夹绝对地址根据自身情况填写，无需加入版本号
 #    rest_api_port为端口号，model_name自定义（后面会用到），model_base_path你保存模型的路径。

