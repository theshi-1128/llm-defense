# llm-defense
This repository contains official implementation of our paper "[Figure it Out: Analyzing-based Jailbreak Attack on Large Language Models](https://arxiv.org/pdf/2407.16205)"

[![arXiv: paper](https://img.shields.io/badge/arXiv-paper-red.svg)](https://arxiv.org/abs/2407.16205)
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

- (**2024/07/31**) We have released the official code of ABJ-Attack!
- (**2024/07/23**) Our paper is on arXiv! Check it out [here](https://arxiv.org/abs/2407.16205)!
- (**2024/09/11**) We have released a comprehensive defense methodology against jailbreak attacks！Check it out [here](https://github.com/theshi-1128/llm-defense)!


## Overview

This repository shares the code of our latest work on LLMs jailbreaking. In this work:

- We further explore the boundary of jailbreak attacks on LLMs and propose ABJ, the first jailbreak attack method specifically designed to assess LLMs’ safety in handling analyzing-based tasks. ABJ generalizes jailbreak attack prompts in two steps: data preparation and data analysis.
- We conduct comprehensive experiments on both opensource (Llama-3, Qwen-2, GLM-4) and closed-source (GPT-3.5-turbo, GPT-4-turbo, Claude-3) LLMs. The results demonstrate that ABJ exhibits exceptional attack effectiveness and efficiency, achieving 94.8% ASR on GPT-4-turbo, while the AE is around 1.
- We show the robustness of ABJ when facing different defense strategies, indicating that mitigating this attack might be difficult. Furthermore, by transforming and modifying the ABJ method, we can enable more stealthy and effective jailbreak attacks on a wider range of harmful scenarios, extending beyond the limitations of finite datasets, which makes it more difficult to defend against. Notably, the modified ABJ has achieved over 85% ASR on Llama-3 and Claude-3, which are considered two of the most secure LLMs by far.

<p align="center">
  <img src="ABJ.png" width="900"/>
</p>


## Argument Specification

- `attack_method`: We implement `4` kind of ABJ Attack, including `original_ABJ`, `modified_ABJ`, `code_based_ABJ`,`adversarial_ABJ`.
  
- `target_model`: The name of target model, including `gpt3`, `gpt4`, `claude3_haiku`, `llama3`, `glm4`, `qwen2`.
  
- `attack_rounds`: Number of iteration rounds, default is `3`.
  
- `target_model_cuda_id`: Number of the GPU for target model, default is `cuda:0`.

  
## Quick Start

1. Clone this repository:

   ```sh
   git clone https://github.com/theshi-1128/ABJ-Attack.git
   ```

2. Build enviroment:

   ```sh
   cd ABJ-Attack
   conda create -n ABJ python==3.10
   conda activate ABJ
   pip install -r requirements.txt
   ```

3. Run ABJ-Attack:

     ```sh
     python ABJ.py \
     -- attack_method [ATTACK METHOD] \
     -- target_model [TARGET MODEL] \
     -- attack_rounds [ATTACK ROUNDS] \
     -- target_model_cuda_id [CUDA ID]
     ```

    For example, to run `original_ABJ` with `gpt-4-turbo-2024-04-09` as the target model on `CUDA:0` for `3` rounds, run
  
     ```sh
     python ABJ.py \
     -- attack_method original_ABJ \
     -- target_model gpt4 \
     -- attack_rounds 3 \
     -- target_model_cuda_id cuda:1
     ```



## Citation

If you find this work useful in your own research, please feel free to leave a star⭐️ and cite our paper:

```bibtex
@article{lin2024figure,
  title={Figure it Out: Analyzing-based Jailbreak Attack on Large Language Models},
  author={Lin, Shi and Li, Rongchang and Wang, Xun and Lin, Changting and Xing, Wenpeng and Han, Meng},
  journal={arXiv preprint arXiv:2407.16205},
  year={2024}
}
