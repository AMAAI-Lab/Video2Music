# Video2Music: Suitable Music Generation from Videos using an Affective Multimodal Transformer model

This repository contains the code and dataset accompanying the paper "Video2Music: Suitable Music Generation from Videos using an Affective Multimodal Transformer model" by Dr. Jaeyong Kang, Prof. Soujanya Poria, and Prof. Dorien Herremans.

- Demo: [https://amaai-lab.github.io/Video2Music/](https://amaai-lab.github.io/Video2Music/)
- Paper: [https://arxiv.org/abs/2311.00968](https://arxiv.org/abs/2311.00968)
- Dataset (MuVi-Sync)
  * MuVi-Sync (features) [(Link)](https://zenodo.org/records/10057093)
  * MuVi-Sync (original video) [(Link)](https://zenodo.org/records/10050294)
  * MuVi-Sync (original audio) [(Link)](https://zenodo.org/records/10062564)
  
## Introduction
We propose a novel AI-powered multimodal music generation framework called Video2Music. This framework uniquely uses video features as conditioning input to generate matching music using a Transformer architecture. By employing cutting-edge technology, our system aims to provide video creators with a seamless and efficient solution for generating tailor-made background music.

![](framework.png)

## Directory Structure

* `saved_models/`: saved model files
* `utilities/`
  * `run_model_vevo.py`: code for running model (AMT)
  * `run_model_regression.py`: code for running model (bi-GRU)
* `model/`
  * `video_music_transformer.py`: Affective Multimodal Transformer (AMT) model 
  * `video_regression.py`: Bi-GRU regression model used for predicting note density/loudness
  * `positional_encoding.py`: code for Positional encoding
  * `rpr.py`: code for RPR (Relative Positional Representation)
* `dataset/`
  * `vevo_dataset.py`: Dataset loader
* `script/` : code for extracting video/music features (sementic, motion, emotion, scene offset, loudness, and note density)
* `train.py`: training script (AMT)
* `train_regression.py`: training script (bi-GRU)
* `evaluate.py`: evaluation script
* `generate.py`: inference script

## Preparation

* Clone this repo

* Obtain the dataset:
  * MuVi-Sync (features) [(Link)](https://zenodo.org/records/10057093)
  * MuVi-Sync (original video) [(Link)](https://zenodo.org/records/10050294)
 
* Put all directories started with `vevo` in the dataset under this folder (`dataset/`) 

* Download the processed training data `AMT.zip` from [HERE](https://drive.google.com/file/d/1qpcBXF04pgdy9hqRexr0mTx7L9_CAFpt/view?usp=drive_link) and extract the zip file and put the extracted two files directly under this folder (`saved_models/AMT/`) 

* Install dependencies `pip install -r requirements.txt`
  * Our code is built on pytorch version 1.12.1 and Python version 3.7.15 (torch==1.13.1 in the requirements.txt). But you might need to choose the correct version of `torch` based on your CUDA version

## Training

  ```shell
  python train.py
  ```

## Inference

  ```shell
  python generate.py
  ```

## Citation
If you find this resource useful, [please cite the original work](https://arxiv.org/abs/2311.00968):

```bibtex
@article{kang2023video2music,
  title={Video2Music: Suitable Music Generation from Videos using an Affective Multimodal Transformer model},
  author={Kang, Jaeyong and Poria, Soujanya and Herremans, Dorien},
  journal={arXiv preprint arXiv:2311.00968},
  year={2023}
}
```

Kang, J., Poria, S. & Herremans, D. (2023). Video2Music: Suitable Music Generation from Videos using an Affective Multimodal Transformer model. arXiv preprint arXiv:2311.00968.


## Acknowledgements

Our code is based on [Music Transformer](https://github.com/gwinndr/MusicTransformer-Pytorch).


