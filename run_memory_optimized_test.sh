#!/bin/bash

# Quick test for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

echo "Testing memory optimization with backFlush/method.py..."
echo "Memory environment set:"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

python backFlush/method.py | tee logs/memory_test.log