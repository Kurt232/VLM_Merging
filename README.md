# VLM_Merging

This repository contains the official implementation for the ICML 2025 paper "Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging".

## Overview

VLM_Merging explores techniques for merging large vision-language models (VLMs) to enhance their capabilities in perception and reasoning tasks. The project leverages model merging strategies to combine the strengths of different VLMs and Math LLMs, resulting in improved performance across various benchmarks.

## Key Features

- Implementation of various model merging techniques including:
  - Base merging
  - Layer swapping
  - TIES merging
  - DARE-TIES and DARE-linear merging

- Evaluation framework using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for comprehensive assessment of merged models

## Installation

```bash
git clone https://github.com/shiqichen17/VLM_Merging.git

cd VLM_Merging

conda create -n vlm_merging python=3.10 -y

conda activate vlm_merging

pip install -r requirements.txt

cd VLMEvalKit

pip install -e .
```

## Usage

### Model Merging

The main merging functionality is provided in `merge.py`. You can merge two models using the following command:

```bash
python merge.py --model1_path /path/to/model1 \
--model2_path /path/to/model2 \
--output_dir /path/to/output \
--alpha 0.5 \
--mode base
```

### Merging Modes

- `base`: Standard weighted averaging of model parameters
- `layerswap`: Swap specific layers between models
- `ties`: Task vector merging using TIES
- `dareties`: DARE-TIES merging with sparse task vectors
- `darelinear`: DARE-linear merging with sparse task vectors

### Additional Parameters

- `--alpha`: Weighting factor for the merge (between 0 and 1)
- `--base_layer_num`: Base layer number (required for layerswap mode)
- `--basemodel_path`: Base model path (required for TIES-based modes)
- `--density`: Sparsity parameter for DARE modes (default: 0.2)
- `--alpha2`: Secondary alpha parameter for some merging modes (default: 0.2)

### Example Scripts

Example scripts for model merging and evaluation are provided in:
- `scripts/merge/example.sh`: Contains examples of different merging strategies
- `scripts/eval/evaluate_vlm.sh`: Shows how to evaluate merged models on various benchmarks

## Evaluation

The repository integrates with VLMEvalKit for comprehensive evaluation of merged models on various benchmarks. See the VLMEvalKit documentation for details on running evaluations.

## Citation

If you find this work helpful, please consider citing our paper.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgements

- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for providing the evaluation framework
- The authors of the original models used in our experiments