# llm-defense
This is a repository for fortifying the security of Large Language Models (LLMs) against jailbreak attacks. If you find this work useful in your own research, please feel free to leave a star⭐️!

![Jailbreak Attacks](https://img.shields.io/badge/Jailbreak-Attacks-yellow.svg?style=plastic)
![Large Language Models](https://img.shields.io/badge/LargeLanguage-Models-green.svg?style=plastic)
[![license: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Please feel free to contact linshizjsu@gmail.com if you have any questions.

## Table of Contents

- [Updates](#updates)
- [Overview](#overview)
- [Argument Specification](#argument-specification)
- [Quick Start](#quick-start)


## Updates

- (**2024/09/11**) We have released a comprehensive defense methodology against jailbreak attacks！


## Overview

This repository shares the code of our work on defending LLMs against jailbreak attacks. We achieve a comprehensive defense pipeline against jailbreak attacks to LLMs without affecting their helpfulness. Specifically, we have implemented `8` different defense methods that cover the entire process of LLMs' inference stage, and we have combined these methods using an expert voting approach, which can effectively increase the success rate of defenses and reduce cases of over-protection.


## Argument Specification
  
- `target_model`: The name of target model.
    
- `target_model_cuda_id`: Number of the GPU for target model, default is `cuda:0`.

- `batch_size`: Number of prompts to process per batch, default is `1`.

  
## Quick Start

Before you start, you should replace the necessary information in `llm/models.py`.


1. Clone this repository:

   ```sh
   git clone https://github.com/theshi-1128/llm-defense.git
   ```

2. Build enviroment:

   ```sh
   cd llm-defense
   conda create -n defense python==3.10
   conda activate defense
   pip install -r requirements.txt
   ```

3. Run llm-defense:

     ```sh
     python main.py \
     -- target_model [TARGET MODEL] \
     -- target_model_cuda_id [CUDA ID]
     -- batch_size [BATCH SIZE]
     ```

    For example, to use `glm-4-9b-chat` as the target model on `CUDA:0` with batch_size of `4`, run
  
     ```sh
     python main.py \
     -- target_model glm4 \
     -- target_model_cuda_id cuda:0
     -- batch_size 4
     ```
     

## Acknowledgement

We have partly leveraged some code from [llm-jailbreaking-defense](https://github.com/YihanWang617/llm-jailbreaking-defense) and [llm-guard](https://github.com/protectai/llm-guard).

We have also referred to code from official implementations of existing methods:
* [GCG](https://github.com/llm-attacks/llm-attacks)
* [ABJ](https://github.com/theshi-1128/ABJ-Attack)
* [LLM-as-a-Judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)
* [IA](https://github.com/alphadl/SafeLLM_with_IntentionAnalysis?tab=readme-ov-file)

