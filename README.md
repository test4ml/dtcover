# DTCover

## Purpose
we propose a novel approach called **DTCover** for Decoder-only Transformers Coverage Guided Adversarial Testing. The main idea of our approach is to design coverage metrics around the modules in the transformer and perform coverage-guided adversarial testing against uncovered neurons.

In addition, we provide an [appendix](DTCover_appendix.pdf) that describes how the three neuron coverage metrics in our algorithm calculate the overall coverage value, as well as the original accuracy of the three large language models on three datasets, and the overlap between adversarial samples tested by different neuron coverage metrics.
In the discussion section of the appendix, we discuss the growth of the three neuron coverage metrics as the sample size increases. Finally, we discuss the time efficiency of DTCover.

## Provenance
This is the source code for the paper "DTCover: Coverage Guided Adversarial Testing for Decoder-only Transformer Language Models"

# Data
Our data and models are from [HuggingFace](https://huggingface.co/), and the code will automatically download the required data and models (Internet connection is required).

# Setup
## Hardware
Hardware Requirements.

Minimum:
```
Requires a 64-bit processor and operating system.
Operating System: Linux distributions.
Processor: Intel Core i5-6600.
Memory: 64 GB RAM.
GPU: NVIDIA RTX A6000. (GPUs are used for Large Language Model inference, requires at least 48GB of graphics memory. If you Large Language Model on the CPU, it may take a significant amount of time.)
Network: Broadband internet connection.
Storage: Requires 128 GB of available space.
```

Tested Hardware:
```
CPU: two slots of 16 Core Intel Xeon Gold 6226R CPU 2.90GHz Processor
Memory: 8x32GB DDR4 DIMM 2933MHz Memory
GPUs: NVIDIA A100 80G GPU.
```

## Software

Tested System:
* 64-bit Ubuntu 22.04.2 LTS Linux kernel 6.2.0

Software Requirements:
* Anaconda3-2023.09-0-Linux-x86_64 (or Miniconda)

Python Requirements:
* tokenizers==0.15.1
* transformers==4.37.2
* accelerate==0.21.0
* torchmetrics==1.2.0
* seaborn==0.13.2
* pydantic==2.4.2
* textattack==0.3.9
* spacy==3.7.2
* einops

All code in this repository is tested under the environment of `Python 3.9.18`. We use conda to construct a virtual environment to run the python program.

## Setup Python Environment
This step requires Conda installation.
```bash
conda create --prefix=dtcoverenv python=3.9.18 --yes
eval "$(conda shell.bash hook)"
conda init
conda activate ./dtcoverenv
pip install -r requirements.txt
```

# Usage

## RQ1: Method Effectiveness
```bash
bash attack.sh
python rqs/summary.py
```
Then the adversarial testing results will be saved in `rq_results/summary.csv`.

## RQ2: Metric Effectiveness
```bash
bash attack_metrics.sh
python rqs/summary_metrics.py
```
Then the adversarial testing results will be saved in `rq_results/summary_metrics.csv`.

## RQ3: Parameter Sensitivity
```bash
bash search_grid.sh
python rqs/draw_search_threshold.py
```
Then the search results will be saved in `figures/search_grid.pdf`.

## RQ4: Model Retraining
Generate adversarial testing results in the valid dataset.
```bash
bash attack_retrain_valid_baselines.sh
```

Collect and generate the retraining dataset.
```bash
python scripts/generate_retrain_baselines_dataset.py
```

Then the retraining dataset will be saved under the `retrain_baselines_dataset`.
We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to retrain these three models.
After retraining, move the models to the path `./retrain_models/Llama-2-7b-chat-hf-${recipe}/`.

```bash
bash attack_retrain_baselines.sh
```
Then the adversarial testing results of retrained models will be saved in `results_retrain/results_${recipe}`.

# LICENSE
Apache License Version 2.0
