# [SBT-Net: A Tri-Cue Guided Multimodal Fusion Framework for Depression Recognition](http://)

[**Yujie Huo, Weng Howe Chan, Ahmad Najmi Bin Amerhaider Nuar and Hongyu Gao**](http://)


## 📰 News
* **[2025-11]**: 🎉 Our paper "SBT-Net: A Tri-Cue Guided Multimodal Fusion Framework for Depression Recognition" has been accepted by **BioData Mining**!
This project implements a multimodal depression prediction model using:
- Text (ALBERT)
- Audio (Wav2Vec2)
- Semantic-Guided Gating (SGCMG)
- Bias-Guided Tensor Product Attention (BG-TPA)
- Emotion Trend Module (ETM)
## Clone demo code
```text
cd /workspace
git clone https://github.com/ghy-yhg/SBT-Net
```
## Dataset Setup

The DAIC-WOZ dataset needs to be used by logging into the official  [website](https://dcapswoz.ict.usc.edu/) and filling out an application form.
Unzip the files and place them as follows:


## 📦 Dataset Format

Expected structure (for both DAIC-WOZ and EATD):

```
data/
├── daic_woz/
│   ├── audio/            
│   ├── transcripts/        
│   └── labels.csv
├── eatd/
│   ├── audio/
│   ├── transcripts/
│   └── labels.csv
```
## Config Introduction
```text
pip install -r requirements.txt
```
### labels.csv format:

```
session_id,text_path,audio_path,label
P3001,transcripts/P3001.txt,audio/P3001.wav,1
...
```

## 🚀 Train

```bash
python train.py
```

## 🧪 Test

```bash
python test.py


```
## Issue
If there is an issue, please send an email to this address，huoyujie@graduate.utm.my

##  Citation
If this repository is helpful for your research, we'd really appreciate it if you could cite the following paper:

```

@article{Huo2025SBTNet,
  author    = {Yujie Huo and Weng Howe Chan and Ahmad Najmi Bin Amerhaider Nuar and Hongyu Gao},
  title     = {SBT-Net: A Tri-Cue Guided Multimodal Fusion Framework for Depression Recognition},
  journal   = {BioData Mining},
  year      = {2025},
  publisher = {Springer},
  note      = {Accepted for publication}
}
```
