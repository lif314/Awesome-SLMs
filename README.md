# Awesome-SLMs [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) [![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com) ![Github Last Commit](https://img.shields.io/github/last-commit/lif314/Awesome-SLMs)


## 🏠 About
This repo contains a curative list of **Small (Vision)-Language Models**. This is an active repository, you can watch for following the latest advances. If you find this repository useful, please consider STARing ⭐ this list. Feel free to share this list with others!


<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#SurveyCollection">Survey/Collection</a>
    </li>
    <li>
      <a href="#Blogs">Blogs</a>
    </li>
    <li>
      <a href="#Insights">Insights</a>
    </li>
    <li>
      <a href="#SmallLMs">SmallLMs</a>
    </li>
    <li>
      <a href="#SmallVLMs">SmallVLMs</a>
    </li>
    <li>
      <a href="#SmallVLAs">SmallVLAs</a>
    </li>
     <li>
      <a href="#SpatialVLMs">SpatialVLMs</a>
    </li>
  </ol>
</details>

## Survey/Collection
- Awesome-SLM: a curated list of Small Language Model. [Github](https://github.com/ro-ko/Awesome-SLM) ![Stars](https://img.shields.io/github/stars/ro-ko/Awesome-SLM?style=social) ![Last Commit](https://img.shields.io/github/last-commit/ro-ko/Awesome-SLM)
- Awesome Small Language Models. [Github](https://github.com/slashml/awesome-small-language-models) ![Stars](https://img.shields.io/github/stars/slashml/awesome-small-language-models?style=social) ![Last Commit](https://img.shields.io/github/last-commit/slashml/awesome-small-language-models)
- Domain-Specific Small Language Models Book. [Github](https://github.com/virtualramblas/Domain-Specific-Small-Language-Models) ![Stars](https://img.shields.io/github/stars/virtualramblas/Domain-Specific-Small-Language-Models?style=social) ![Last Commit](https://img.shields.io/github/last-commit/virtualramblas/Domain-Specific-Small-Language-Models)
- Survery of Small Language Models. [Github](https://github.com/OpenCSGs/Awesome-SLMs) ![Stars](https://img.shields.io/github/stars/OpenCSGs/Awesome-SLMs?style=social) ![Last Commit](https://img.shields.io/github/last-commit/OpenCSGs/Awesome-SLMs)
- Small VLM - a bunnycore Collection. [HF Collection](https://huggingface.co/collections/bunnycore/small-vlm)
- Phi Cookbook: Hands-On Examples with Microsoft's Phi Models. [Github](https://github.com/microsoft/PhiCookBook) ![Stars](https://img.shields.io/github/stars/microsoft/PhiCookBook?style=social) ![Last Commit](https://img.shields.io/github/last-commit/microsoft/PhiCookBook)
- Tiny VLMs Lab: interactive web application. [Github](https://github.com/PRITHIVSAKTHIUR/Tiny-VLMs-Lab) ![Stars](https://img.shields.io/github/stars/PRITHIVSAKTHIUR/Tiny-VLMs-Lab?style=social) ![Last Commit](https://img.shields.io/github/last-commit/PRITHIVSAKTHIUR/Tiny-VLMs-Lab)


## Blogs
- 垂直领域 Agent 落地：为什么我放弃 235B/671B，转而训练 8B. [Zhihu (一)](https://zhuanlan.zhihu.com/p/1994124775983970078) [Zhihu (二)](https://zhuanlan.zhihu.com/p/1994409280238012063)
- 小模型层数好玄学：12/32/64层效果好，16/24/48/层效果糟. [Zhihu](https://zhuanlan.zhihu.com/p/1993674796538565507) [Source](https://huggingface.co/blog/codelion/optimal-model-architecture) `70M`
- qwen3-0.6B这种小模型有什么实际意义和用途吗？ [Zhihu](https://www.zhihu.com/question/1900664888608691102)
- Small Language Models are the Future of Agentic AI, NVIDIA Research. [Paper](https://arxiv.org/abs/2506.02153), [Website](https://research.nvidia.com/labs/lpr/slm-agents/), [Blog](https://developer.nvidia.com/blog/how-small-language-models-are-key-to-scalable-agentic-ai/)
- Small Language Models (SLM): A Comprehensive Overview. [Blog](https://huggingface.co/blog/jjokah/small-language-model)
- What are small language models? IBM Thinker. [Blog](https://www.ibm.com/think/topics/small-language-models)
- What Are Small Language Models (SLMs)? Microsoft Cloud Team. [Blog](https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/what-are-small-language-models)
- The Best Open-Source Small Language Models (SLMs) in 2026. [BentoML Blog](https://www.bentoml.com/blog/the-best-open-source-small-language-models)


## Insights
- An Information Theoretic Perspective on Agentic System Design, arXiv 2025. [Paper](https://www.arxiv.org/abs/2512.21720) `SLMs are all you need in Agent system.`
- :fire: Dhara-70M: The Optimal Architecture for Small Language Models. `70M` [Blog](https://huggingface.co/blog/codelion/optimal-model-architecture) [Model](https://huggingface.co/codelion/dhara-70m) `For SLMs, the depth-to-width ratio is more important than the model architecture.`
- Return of the Encoder: Efficient Small Language Models, arXiv 2025. [Paper](https://arxiv.org/pdf/2501.16273), [Github](https://github.com/microsoft/encoder-decoder-slm) ![Stars](https://img.shields.io/github/stars/microsoft/encoder-decoder-slm?style=social) ![Last Commit](https://img.shields.io/github/last-commit/microsoft/encoder-decoder-slm) `Encoder-decoder models inherently outperform decoder-only architectures before any optimizations.`

## SmallLMs
> [Github]() ![Stars](https://img.shields.io/github/stars/?style=social) ![Last Commit](https://img.shields.io/github/last-commit/)
- NanoChat: The best ChatGPT that $100 can buy. `2.2B`[Github](https://github.com/karpathy/nanochat) ![Stars](https://img.shields.io/github/stars/karpathy/nanochat?style=social) ![Last Commit](https://img.shields.io/github/last-commit/karpathy/nanochat)
- NanoGPT: The simplest, fastest repository for training/finetuning medium-sized GPTs. `124M` [Github](https://github.com/karpathy/nanoGPT) ![Stars](https://img.shields.io/github/stars/karpathy/nanoGPT?style=social) ![Last Commit](https://img.shields.io/github/last-commit/karpathy/nanoGPT)
- MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes, arXiv 2025, Facebook. `140M/360M/950M` [Paper](https://arxiv.org/abs/2509.24945) [Model](https://huggingface.co/collections/facebook/mobilellm-r1) [Github](https://github.com/facebookresearch/MobileLLM-R1) ![Stars](https://img.shields.io/github/stars/facebookresearch/MobileLLM-R1?style=social) ![Last Commit](https://img.shields.io/github/last-commit/facebookresearch/MobileLLM-R1)
- MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases, ICML 2024. Facebook.  `125M-1.5B` [Paper](https://arxiv.org/abs/2402.14905) [Model](https://huggingface.co/collections/facebook/mobilellm) [Github](https://github.com/facebookresearch/MobileLLM) ![Stars](https://img.shields.io/github/stars/facebookresearch/MobileLLM?style=social) ![Last Commit](https://img.shields.io/github/last-commit/facebookresearch/MobileLLM)

- Qwen3. `0.6B-235B` [Paper](https://arxiv.org/abs/2505.09388) [Github](https://github.com/QwenLM/Qwen3) ![Stars](https://img.shields.io/github/stars/QwenLM/Qwen3?style=social) ![Last Commit](https://img.shields.io/github/last-commit/QwenLM/Qwen3)
- Building a Small Language Model(SLM) from Scratch. [Github](https://github.com/ChaitanyaK77/Building-a-Small-Language-Model-SLM-) ![Stars](https://img.shields.io/github/stars/ChaitanyaK77/Building-a-Small-Language-Model-SLM-?style=social) ![Last Commit](https://img.shields.io/github/last-commit/ChaitanyaK77/Building-a-Small-Language-Model-SLM-)
- SmallDoge: Ultra-Fast Small Language Models, `20M-320M`, [Github](https://github.com/SmallDoges/small-doge) ![Stars](https://img.shields.io/github/stars/SmallDoges/small-doge?style=social) ![Last Commit](https://img.shields.io/github/last-commit/SmallDoges/small-doge)  
- SmolLM2 (Language Model), `135M-1.7B`, [Github](https://github.com/huggingface/smollm) ![Stars](https://img.shields.io/github/stars/huggingface/smollm?style=social) ![Last Commit](https://img.shields.io/github/last-commit/huggingface/smollm)
- SmolLM3 (Language Model), `3B`, [Github](https://github.com/huggingface/smollm) ![Stars](https://img.shields.io/github/stars/huggingface/smollm?style=social) ![Last Commit](https://img.shields.io/github/last-commit/huggingface/smollm)
- SLM-SQL: An Exploration of Small Language Models for Text-to-SQL. [Paper](https://arxiv.org/abs/2507.22478), [Github](https://github.com/CycloneBoy/slm_sql) ![Stars](https://img.shields.io/github/stars/CycloneBoy/slm_sql?style=social) ![Last Commit](https://img.shields.io/github/last-commit/CycloneBoy/slm_sql)
- Tiny language models (pre-training and fine-tuning a compact BERT model ). [Paper](https://arxiv.org/abs/2507.14871),[Github](https://github.com/Rg32601/Tiny-Language-Models) ![Stars](https://img.shields.io/github/stars/Rg32601/Tiny-Language-Models?style=social) ![Last Commit](https://img.shields.io/github/last-commit/Rg32601/Tiny-Language-Models)



##  SmallVLMs
> [Github]() ![Stars](https://img.shields.io/github/stars/?style=social) ![Last Commit](https://img.shields.io/github/last-commit/)
- NVILA: Optimized Vision Language Models, NVIDIA. `2B-15B`. [Model](https://huggingface.co/collections/Efficient-Large-Model/nvila) [Github](https://github.com/NVlabs/VILA) ![Stars](https://img.shields.io/github/stars/NVlabs/VILA?style=social) ![Last Commit](https://img.shields.io/github/last-commit/NVlabs/VILA)
- Fine-tuning Qwen-VL Series. [Github](https://github.com/2U1/Qwen-VL-Series-Finetune) ![Stars](https://img.shields.io/github/stars/2U1/Qwen-VL-Series-Finetune?style=social) ![Last Commit](https://img.shields.io/github/last-commit/2U1/Qwen-VL-Series-Finetune)
-  Qwen3-Embedding and Qwen3-Reranker. `0.6B-8B`  [Github](https://github.com/QwenLM/Qwen3-VL-Embedding) ![Stars](https://img.shields.io/github/stars/QwenLM/Qwen3-VL-Embedding?style=social) ![Last Commit](https://img.shields.io/github/last-commit/QwenLM/Qwen3-VL-Embedding)
- Qwen3+SmolVLM: 将SmolVLM2的视觉头与Qwen3-0.6B模型进行了拼接微调. [Github](https://github.com/ShaohonChen/Qwen3-SmVL) ![Stars](https://img.shields.io/github/stars/ShaohonChen/Qwen3-SmVL?style=social) ![Last Commit](https://img.shields.io/github/last-commit/ShaohonChen/Qwen3-SmVL)
- Smol Vision: Recipes for shrinking, optimizing, customizing cutting edge vision and multimodal AI models. [Github](https://github.com/merveenoyan/smol-vision) ![Stars](https://img.shields.io/github/stars/merveenoyan/smol-vision?style=social) ![Last Commit](https://img.shields.io/github/last-commit/merveenoyan/smol-vision)
- SmolVLM (Vision Language Model), `236M-2B`, [Github](https://github.com/huggingface/smollm) ![Stars](https://img.shields.io/github/stars/huggingface/smollm?style=social) ![Last Commit](https://img.shields.io/github/last-commit/huggingface/smollm)
- nanoVLM: The simplest repository to train your VLM in pure PyTorch, `222M`, [Github](https://github.com/huggingface/nanoVLM) ![Stars](https://img.shields.io/github/stars/huggingface/nanoVLM?style=social) ![Last Commit](https://img.shields.io/github/last-commit/huggingface/nanoVLM)
- NanoVLM-Lab: Democratizing Vision-Language Model Training for Everyone, `222M`, [Github](https://github.com/akash-kamalesh/nanovlm-lab) ![Stars](https://img.shields.io/github/stars/akash-kamalesh/nanovlm-lab?style=social) ![Last Commit](https://img.shields.io/github/last-commit/akash-kamalesh/nanovlm-lab)
- Fine-tuning Florence-2 - Microsoft's Cutting-edge Vision Language Models, `0.2B/0.7B`, [Github](https://github.com/andimarafioti/florence2-finetuning) ![Stars](https://img.shields.io/github/stars/andimarafioti/florence2-finetuning?style=social) ![Last Commit](https://img.shields.io/github/last-commit/andimarafioti/florence2-finetuning)
- Bunny: A family of lightweight multimodal models. [Paper](https://arxiv.org/abs/2402.11530), [Github](https://github.com/BAAI-DCAI/Bunny) ![Stars](https://img.shields.io/github/stars/BAAI-DCAI/Bunny?style=social) ![Last Commit](https://img.shields.io/github/last-commit/BAAI-DCAI/Bunny)
- Moondream: a tiny vision language model that kicks ass and runs anywhere. `0.5B/2B`, [Github](https://github.com/vikhyat/moondream) ![Stars](https://img.shields.io/github/stars/vikhyat/moondream?style=social) ![Last Commit](https://img.shields.io/github/last-commit/vikhyat/moondream)
- FastVLM: Efficient Vision Encoding for Vision Language Models, CVPR 2025. `0.5B/1.5B/7B`, [Paper](https://www.arxiv.org/abs/2412.13303), [Github](https://github.com/apple/ml-fastvlm) ![Stars](https://img.shields.io/github/stars/apple/ml-fastvlm?style=social) ![Last Commit](https://img.shields.io/github/last-commit/apple/ml-fastvlm)
- From Pixels to Words -- Towards Native Vision-Language Primitives at Scale, arXiv 2025. `2B/9B`, [Paper](https://arxiv.org/abs/2510.14979),  [Github](https://github.com/EvolvingLMMs-Lab/NEO) ![Stars](https://img.shields.io/github/stars/EvolvingLMMs-Lab/NEO?style=social) ![Last Commit](https://img.shields.io/github/last-commit/EvolvingLMMs-Lab/NEO)

**Qwen-VL Series**
- **Qwen-VL**: Initial large vision-language model with multilingual support, grounding, and fine-grained understanding. `7B`, [Paper](https://arxiv.org/abs/2308.12966), [Github](https://github.com/QwenLM/Qwen-VL) ![Stars](https://img.shields.io/github/stars/QwenLM/Qwen-VL?style=social) ![Last Commit](https://img.shields.io/github/last-commit/QwenLM/Qwen-VL)

- **Qwen2-VL**: Enhanced perception at any resolution with dynamic resolution support. `2B/7B/72B`, [Paper](https://arxiv.org/abs/2409.12191), [Model](https://huggingface.co/collections/Qwen/qwen2-vl), [Github](https://github.com/QwenLM/Qwen3-VL)

- **Qwen2.5-VL**: Flagship upgrade with improved visual recognition, object localization, document parsing, long-video understanding, and visual agent capabilities. `3B/7B/32B/72B`, [Paper](https://arxiv.org/abs/2502.13923), [Model](https://huggingface.co/collections/Qwen/qwen25-vl), [Github](https://github.com/QwenLM/Qwen3-VL) ![Stars](https://img.shields.io/github/stars/QwenLM/Qwen3-VL?style=social) ![Last Commit](https://img.shields.io/github/last-commit/QwenLM/Qwen3-VL)

- **Qwen3-VL**: Latest generation with comprehensive upgrades in vision, reasoning, and multimodal capabilities (including thinking modes). `2B-235B`, [Blog](https://qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef), [Model](https://huggingface.co/collections/Qwen/qwen3-vl), [Github](https://github.com/QwenLM/Qwen3-VL) ![Stars](https://img.shields.io/github/stars/QwenLM/Qwen3-VL?style=social) ![Last Commit](https://img.shields.io/github/last-commit/QwenLM/Qwen3-VL)



**LLaVA Series**
- **LLaVA**: Large Language and Vision Assistant. `7B/13B`, [Paper](https://arxiv.org/abs/2304.08485), [Github](https://github.com/haotian-liu/LLaVA) ![Stars](https://img.shields.io/github/stars/haotian-liu/LLaVA?style=social) ![Last Commit](https://img.shields.io/github/last-commit/haotian-liu/LLaVA)

- **LLaVA-1.5**: Improved LLaVA with MLP projector and better training data. `7B/13B`, [Paper](https://arxiv.org/abs/2310.03744), [Github](https://github.com/haotian-liu/LLaVA) ![Stars](https://img.shields.io/github/stars/haotian-liu/LLaVA?style=social) ![Last Commit](https://img.shields.io/github/last-commit/haotian-liu/LLaVA)

- **LLaVA-NeXT (LLaVA-1.6)**: Enhanced reasoning, OCR, world knowledge, higher resolution support. `7B/8B/13B/34B/72B/110B`, [Blog/Paper](https://llava-vl.github.io/blog/2024-01-30-llava-next/), [Github](https://github.com/LLaVA-VL/LLaVA-NeXT) ![Stars](https://img.shields.io/github/stars/LLaVA-VL/LLaVA-NeXT?style=social) ![Last Commit](https://img.shields.io/github/last-commit/LLaVA-VL/LLaVA-NeXT)

- **MobileVLM v1/v2**: Vision Language Model for Mobile Devices. `1.4B/2.7B` (v1), `1.7B/3B` (v2), [Paper v1](https://arxiv.org/abs/2312.16886) / [Paper v2](https://arxiv.org/abs/2402.03766), [Github](https://github.com/Meituan-AutoML/MobileVLM) ![Stars](https://img.shields.io/github/stars/Meituan-AutoML/MobileVLM?style=social) ![Last Commit](https://img.shields.io/github/last-commit/Meituan-AutoML/MobileVLM)

- **TinyLLaVA**: Framework for small-scale LLaVA models, focusing on lightweight deployment. `0.5B-3B`, [Paper](https://arxiv.org/abs/2402.14289), [Github](https://github.com/TinyLLaVA/TinyLLaVA_Factory) ![Stars](https://img.shields.io/github/stars/TinyLLaVA/TinyLLaVA_Factory?style=social) ![Last Commit](https://img.shields.io/github/last-commit/TinyLLaVA/TinyLLaVA_Factory)

- **LLaVA-Mini**: Efficient Image/Video LMM with one vision token compression. `~8B`, [Paper](https://arxiv.org/abs/2501.03895), [Github](https://github.com/ictnlp/LLaVA-Mini) ![Stars](https://img.shields.io/github/stars/ictnlp/LLaVA-Mini?style=social) ![Last Commit](https://img.shields.io/github/last-commit/ictnlp/LLaVA-Mini)

- **MoE-LLaVA**: Mixture-of-Experts based lightweight LLaVA. `2B-7B`, [Paper](https://arxiv.org/abs/2404.03768), [Github](https://github.com/PKU-YuanGroup/MoE-LLaVA)

---

- **SmolVLM / SmolVLM2**: Ultra-efficient family designed for consumer devices, including video understanding in tiny sizes. `256M/500M/2.2B`, [Blog SmolVLM](https://huggingface.co/blog/smolvlm), [Blog SmolVLM2](https://huggingface.co/blog/smolvlm2), [Model Collection](https://huggingface.co/collections/HuggingFaceTB/smolvlm2-smallest-video-lm-ever-67ab6b5e84bf8aaa60cb17c7)

- **DeepSeek-VL**: MoE-based lightweight VLM excelling in reasoning and scientific tasks. `1.3B`, [Paper](https://arxiv.org/abs/2403.05525), [Model](https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-chat) [Github](https://github.com/deepseek-ai/DeepSeek-VL) 

- **MiniCPM-V**: Efficient multimodal model strong in OCR, charts, and mobile/edge deployment. `4B-12B`, [Model](https://huggingface.co/openbmb/MiniCPM-V-2_6) [Github](https://github.com/OpenBMB/MiniCPM-V)


- **Phi-3-Vision / Phi-4 Multimodal**: Microsoft's small efficient VLM with strong OCR and reasoning. `~3B-4B` (vision variants), [Paper](https://arxiv.org/abs/2404.14219), [Model](https://huggingface.co/collections/microsoft/phi-3-vision)

- **PaliGemma**: Fine-tunable small VLM with SigLIP encoder. `3B` (smallest variant), [Paper](https://arxiv.org/abs/2407.07726), [Model](https://huggingface.co/google/paligemma-3b-mix-224)

- **Ministral 3B**: Mistral AI's edge-focused multimodal model with image understanding, reasoning, and efficient on-device deployment. `3B`, [Blog](https://mistral.ai/news/ministral-3), [Model](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512)

- **Moondream2**: Compact vision-language model excels in detailed image description and VQA, highly efficient for mobile/edge. `~1.8B-2B`, [Github](https://github.com/vikhyat/moondream), [Model](https://huggingface.co/vikhyatk/moondream2)

- **Molmo-1B**: Allen AI's open-source VLM with strong performance in visual reasoning and understanding, competitive with larger models. `1B`, [Blog](https://molmo.allenai.org/blog), [Model](https://huggingface.co/allenai/molmo-1b)

- **Gemma 3 Multimodal (small variants)**: Google's lightweight multimodal extension with high-res support and long context. `~2B-4B` (effective small), [Blog](https://deepmind.google/technologies/gemma/), [Model](https://huggingface.co/google/gemma-3-multimodal)

- **PaddleOCR-VL**: Ultra-lightweight OCR-focused VLM with dynamic resolution for document parsing. `0.9B`, [Paper](https://arxiv.org/abs/2510.14528), [Github](https://github.com/PaddlePaddle/PaddleOCR)

- **dots.ocr**: Specialized small VLM fine-tuned for multilingual OCR and complex document understanding. `3B`, [Github](https://github.com/rednote-hilab/dots.ocr), [Model](https://huggingface.co/rednote/dots.ocr)

- **DeepSeek Janus (small variants)**: Efficient multimodal series with strong visual reasoning. `1B/1.3B`, [Model](https://huggingface.co/deepseek-ai/janus-1.3b)

## SmallVLAs
> [Github]() ![Stars](https://img.shields.io/github/stars/?style=social) ![Last Commit](https://img.shields.io/github/last-commit/)
- [Awesome Vision Language Action (VLA) Models](https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln?tab=readme-ov-file#-vision-language-action-vla-models) ![Stars](https://img.shields.io/github/stars/jonyzhang2023/awesome-embodied-vla-va-vln?style=social) ![Last Commit](https://img.shields.io/github/last-commit/jonyzhang2023/awesome-embodied-vla-va-vln)
- SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics, `0.24B-2.25B`, [Paper](https://arxiv.org/abs/2506.01844), [Github](https://github.com/huggingface/lerobot) ![Stars](https://img.shields.io/github/stars/huggingface/lerobot?style=social) ![Last Commit](https://img.shields.io/github/last-commit/huggingface/lerobot)
- X-VLA: Soft-Prompted Transformer as a Scalable Cross-Embodiment Vision-Language-Action Model, `0.9B`, [Paper](https://arxiv.org/pdf/2510.10274), [Github](https://github.com/2toinf/X-VLA) ![Stars](https://img.shields.io/github/stars/2toinf/X-VLA?style=social) ![Last Commit](https://img.shields.io/github/last-commit/2toinf/X-VLA)
- VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model, `0.5B`, [Paper](https://arxiv.org/abs/2509.09372), [Github](https://github.com/OpenHelix-Team/VLA-Adapter) ![Stars](https://img.shields.io/github/stars/OpenHelix-Team/VLA-Adapter?style=social) ![Last Commit](https://img.shields.io/github/last-commit/OpenHelix-Team/VLA-Adapter)
- TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation, R-AL 2025. `400M-1.3B`, [Paper](https://arxiv.org/abs/2409.12514), [Github](https://github.com/liyaxuanliyaxuan/TinyVLA) ![Stars](https://img.shields.io/github/stars/liyaxuanliyaxuan/TinyVLA?style=social) ![Last Commit](https://img.shields.io/github/last-commit/liyaxuanliyaxuan/TinyVLA)

## SpatialVLMs
> [Github]() ![Stars](https://img.shields.io/github/stars/?style=social) ![Last Commit](https://img.shields.io/github/last-commit/)
- DepthLM: Metric Depth From Vision Language Models, arxiv 2025. [Paper](https://arxiv.org/abs/2509.25413) [Model](https://huggingface.co/facebook/DepthLM) [Github](https://github.com/facebookresearch/DepthLM_Official) ![Stars](https://img.shields.io/github/stars/facebookresearch/DepthLM_Official?style=social) ![Last Commit](https://img.shields.io/github/last-commit/facebookresearch/DepthLM_Official)

