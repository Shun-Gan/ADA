# Adaptive-Driver-Attention

## Introduction 

This repository provide the codes of Adaptive Driver Attention (ADA) model to predict salient regions in different traffic scene

![Fig1](./Fig1.png)

## Pytorch Implementation

### Installation

clone this repository

```
git clone https://github.com/Shun-Gan/ADA.git
cd ADA
```

### Requirements

create a new conda environment and install following packages:

```
conda create --name ADA python=3.8
conda activate ADA
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install scipy
conda install opencv
pip install tensorboardX
```

## Inference

We provide a inference demo  for different driver attention dataset, i.e., BDD-A, DADA-2000, DReyeVE, and EyeTrack.

### Pretrained Models

Firstly, download the pretrained weight file at [Google Driver](https://drive.google.com/file/d/1K0jftj32bOeoUWGTx1XPf58WXrR6m-PC/view?usp=sharing) ( put it in "./weights" folder)

### Example files

Then, download the traffic frames from the four datasets at [Google Driver](https://drive.google.com/file/d/1BCBXJffa6rDY4UjMrFxkFnk4wTwVhj9P/view?usp=sharing) (Unzip and put it in root path) 

### Run Demo

Infer the example files,  or you can put any traffic frames in the same path.

```python
python inference.py --write_video
```

