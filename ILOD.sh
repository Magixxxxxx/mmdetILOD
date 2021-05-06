#!/usr/bin/env bash
CONFIG='ILOD/train/ilod_faster_rcnn_r50_fpn_1x_voc.py'
GPUS=4
PORT=23334
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch ${@:3}