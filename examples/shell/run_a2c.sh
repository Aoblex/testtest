#!/bin/bash

python train.py \
    --env CartPole-v1 \
    --algo a2c \
    --episodes 1000 \
    --eval-interval 10 \
    --lr 3e-4 \
    --gamma 0.99 \
    --value-coef 0.5 \
    --entropy-coef 0.01 \
    --seed 42 