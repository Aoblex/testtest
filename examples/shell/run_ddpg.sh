#!/bin/bash

python train.py \
    --env Pendulum-v1 \
    --algo ddpg \
    --episodes 30 \
    --eval-interval 10 \
    --lr 1e-4 \
    --gamma 0.99 \
    --tau 0.005 \
    --seed 42 