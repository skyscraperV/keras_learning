

#mlp
docker run -d -p 8501:8501 --name simple_mlp \
  --mount type=bind,source=/Users/qian/PycharmProjects/keras_learning/simple/model/mlp.pb,target=/models/simple_mlp  \
  -e MODEL_NAME=simple_mlp -t tensorflow/serving

# mlp info
curl http://localhost:8501/v1/models/simple_mlp
curl http://localhost:8501/v1/models/simple_mlp/metadata

# mlp predict
curl -w "\n====time_total: %{time_total}====\n" \
  -d '{"instances": [[10],[30],[3]]}' -X POST http://localhost:8501/v1/models/simple_mlp:predict

# tensorboard
tensorboard --logdir=model/mlp_tfboard_log/1