#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}


GPUS=4
PORT=${PORT:-29500}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_faster.py  ${@:3}
