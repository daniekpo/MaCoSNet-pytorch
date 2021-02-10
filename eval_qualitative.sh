DATASET=butterfly

MODEL_DIR=trained_models

W_MATCH=1.0
W_CYCLE=0.0
W_TRANS=0.0
W_COSEG=0.0
W_TASK=0.0

MODEL_PATH=$MODEL_DIR/model.pth.tar

python eval.py --model $MODEL_PATH