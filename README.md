# RS-LLaVA: Large Vision Language Model for Joint Captioning and Question Answering in Remote Sensing Imagery

Yakoub Bazi, Laila Bashmal, Mohamad Al rahhal, Riccardo Ricci, Farid Melgani

---

## Content 📒
- [Latest Updates](#latest-updates)
- [Architecture](#architecture)
- [RS-Instructions Dataset](#rs-instructions-dataset)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## Latest Updates  
- ⏰ Soon: RS-instruction dataset.
- 📦 04-Apr-2024: Demo released! 🚀
  
---

## Architecture
<p align="center">
  <img width="600" src="images/architecture.png" alt="RS-LLaVA Architectural Overview">
</p>

---

## Model Release

| Model | Link |
| --- | --- |
|Vicuna-7b|[Vicuna-7b](https://huggingface.co/)|
|Vicuna-13b|[Vicuna-13b](https://huggingface.co/)|

---

## Demo

Demo is coming soon.

---
## RS-Instructions Dataset
<p align="center">
  <img width="600" alt="image" src="https://github.com/BigData-KSU/RS-LLaVA/images/RS_instructions_dataset.png">
</p>

The **RS-instructions** dataset is created by combining four captioning and VQA datasets. Specifically, it includes two captioning datasets, [UCM-caption](https://pan.baidu.com/s/1mjPToHq#list/path=%2F) and UAV, as well as two VQA datasets, [RSVQA-LR](https://rsvqa.sylvainlobry.com/), and [RSIVQA-DOTA](https://github.com/spectralpublic/RSIVQA). We have utilized the same training and testing split as the original datasets. As a result, the **RS-instructions** dataset consists of 7,058 samples, with 5,506 samples in the training set and 1,552 samples in the test set.

The VQA datasets have been formatted in a conversational format. On the other hand, the captioning datasets have been transformed into an instruction-answer format using a set of instructions that simply ask for a description of the image, such as "Provide a description of the image" and "What does this image represent?".


| Dataset | File | Size |
| --- | --- | --- |
UCM-caption| UCM_caption_Train.json | 2.00 MB | 
UCM-caption| UCM_caption_Test.json | 1.20 MB | 
UAV        | UAV_Train.json | 1.20 MB | 
UAV        | UAV_Test.json | 1.20 MB | 
RSVQA-LR   | RSVQA_LR_Train.json | 1.20 MB | 
RSVQA-LR   | RSVQA_LR_Test.json | 1.20 MB | 
RSIVQA-DOTA   | RSIVQA_DOTA_Train.json | 1.20 MB | 
RSIVQA-DOTA   | RSIVQA_DOTA_Test.json | 1.20 MB | 



---

## Acknowledgements
+ [LLaVA](https://github.com/haotian-liu/LLaVA).
+ [Vicuna](https://github.com/lm-sys/FastChat).

---

## Citation

```bibtex
soon

```
---
