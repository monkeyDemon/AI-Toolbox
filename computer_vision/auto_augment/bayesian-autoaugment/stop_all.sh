#!/bin/bash

ps aux | grep "python evaluator.py" | grep -v "grep" | awk '{print $2}' | xargs kill -9
ps aux | grep "python scheduler.py" | grep -v "grep" | awk '{print $2}' | xargs kill -9
ps aux | grep "python evaluator/keras_evaluator/train_and_evaluate.py" | grep -v "grep" | awk '{print $2}' | xargs kill -9
