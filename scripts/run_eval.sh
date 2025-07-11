#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES="3" bash scripts/eval.sh llava_dart_uniform

CUDA_VISIBLE_DEVICES="3" bash scripts/eval.sh llava_deepseek