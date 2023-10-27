# Video2Music: Music Generation to Match Video using an Affective Multimodal Transformer model

This repository contains the code and dataset accompanying the paper "Video2Music: Suitable Music Generation from Videos using an Affective Multimodal Transformer model" by Dr. Jaeyong Kang, Prof. Soujanya Poria, and Prof. Dorien Herremans.

- Demo: [https://amaai-lab.github.io/AIMuVi/](https://amaai-lab.github.io/AIMuVi/)

## Introduction
we propose a novel AI-powered multimodal music generation framework called Video2Music. This framework uniquely uses video features as conditioning input to generate matching music using a Transformer architecture. By employing cutting-edge technology, our system aims to provide video creators with a seamless and efficient solution for generating tailor-made background music.

![](framework.png)

If you find this resource useful, [please cite the original work](https://arxiv.org/abs/XXX):

      @article{XXX,
        title={Music Generation to Match Video using an Affective Multimodal Transformer model},
        author={Kang, Jaeyong and Poria, Soujanya and Herremans, Dorien},
        journal={arXiv preprint arXiv:XXX},
        year={2023}
      }

  Kang, J., Poria, S. & Herremans, D. (2023). Music Generation to Match Video using an Affective Multimodal Transformer model. arXiv preprint arXiv:XXX.


## Directory Structure

* `saved_models/`: code of the whole pipeline
* `utilities/`: code of the whole pipeline
  * `run_model_vevo.py`: training script, take a npz as input music data to train the model
  * `run_model_regression.py`: training script, take a npz as input music data to train the model
* `model/`: code of the whole pipeline
  * `video_music_transformer.py`: training script, take a npz as input music data to train the model
  * `video_regression.py`: training script, take a npz as input music data to train the model
  * `positional_encoding.py`: training script, take a npz as input music data to train the model
  * `rpr.py`: training script, take a npz as input music data to train the model
* `dataset/`: processed dataset for training, in the format of npz
  * `vevo_dataset.py`: training
  * `vevo/` :
* `train.py`: training script, take a npz as input music data to train the model 
* `evaluate.py`: training script, take a npz as input music data to train the model 
* `generate.py`: training script, take a npz as input music data to train the model 


## Preparation

* Clone this repo

* Obtain the dataset:
  * Muvi-Sync [(5 MB)]()
  * Muvi-Sync (audio, video) (option) [(5 MB)]()

* Download the processed training data `AMT.zip` from [HERE](https://drive.google.com/file/d/1ZPQiTyz8wqxwPdYxYSCEtq4MLbR5s9jh/view?usp=drive_link) and extract the zip file and put the extracted two files directly under this folder (`dataset/AMT/`) 

* Install dependencies `pip install -r requirements.txt`
  * Choose the correct version of `torch` based on your CUDA version

## Training

  ```shell
  python train.py
  ```

## Inference

  ```shell
  python generate.py
  ```

