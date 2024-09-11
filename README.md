# llm-defense
This is a repository for fortifying the security of Large Language Models (LLMs) against jailbreak attacks.

![Jailbreak Attacks](https://img.shields.io/badge/Jailbreak-Attacks-yellow.svg?style=plastic)
![Large Language Models](https://img.shields.io/badge/LargeLanguage-Models-green.svg?style=plastic)
[![license: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Please feel free to contact linshizjsu@gmail.com if you have any questions.

## Table of Contents

- [Updates](#updates)
- [Overview](#overview)
- [Quick Start](#quick-start)


## Updates

- (**2024/09/11**) We have released a comprehensive defense methodology against jailbreak attacks！


## Overview

This repository shares the code of our work on defending LLMs against jailbreak attacks. 


## Argument Specification
  
- `target_model`: The name of target model.
    
- `target_model_cuda_id`: Number of the GPU for target model, default is `cuda:0`.

- `batch_size`: Number of prompts to process per batch, default is `1`.

  
## Quick Start

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
