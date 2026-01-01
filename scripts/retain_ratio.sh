#!/bin/bash

python backFlush/method.py --main_ratio 0 --method GA | tee logs/retain_ratio_GA.log
python backFlush/method.py --main_ratio 0.01 --method GA | tee -a logs/retain_ratio_GA.log
python backFlush/method.py --main_ratio 0.02 --method GA | tee -a logs/retain_ratio_GA.log
python backFlush/method.py --main_ratio 0.05 --method GA | tee -a logs/retain_ratio_GA.log
python backFlush/method.py --main_ratio 0.1 --method GA |  tee -a logs/retain_ratio_GA.log
python backFlush/method.py --main_ratio 0.3 --method GA | tee -a logs/retain_ratio_GA.log
python backFlush/method.py --main_ratio 0.5 --method GA | tee -a logs/retain_ratio_GA.log
python backFlush/method.py --main_ratio 1.0 --method GA | tee -a logs/retain_ratio_GA.log

python backFlush/method.py --main_ratio 0.01 --method rot | tee logs/retain_ratio_rot.log
python backFlush/method.py --main_ratio 0.02 --method rot | tee -a logs/retain_ratio_rot.log
python backFlush/method.py --main_ratio 0.05 --method rot | tee -a logs/retain_ratio_rot.log
python backFlush/method.py --main_ratio 0.1 --method rot |  tee -a logs/retain_ratio_rot.log
python backFlush/method.py --main_ratio 0.3 --method rot | tee -a logs/retain_ratio_rot.log
python backFlush/method.py --main_ratio 0.5 --method rot | tee -a logs/retain_ratio_rot.log
python backFlush/method.py --main_ratio 1.0 --method rot | tee -a logs/retain_ratio_rot.log