# Video2Music: Music Generation to Match Video using an Affective Multimodal Transformer model

This repository contains the code and dataset accompanying the paper "Video2Music: Suitable Music Generation from Videos using an Affective Multimodal Transformer model" by Dr. Jaeyong Kang, Prof. Soujanya Poria, and Prof. Dorien Herremans.

- Demo: [https://amaai-lab.github.io/AIMuVi/](https://amaai-lab.github.io/AIMuVi/)
- Dataset

![](framework.png)

**Abstract:**
_Numerous studies in the field of music generation have demonstrated impressive performance, yet virtually no models are able to directly generate music to match accompanying videos. In this work, we develop a generative music AI framework that can match a provided video. We first curated a unique collection of music videos. Then, we analysed the music videos to obtain semantic, scene offset, motion, and emotion features. These distinct features are then employed as guiding input to our music generation model. We transcribe the audio files into MIDI and chords, and extract features such as note density and loudness. This results in a rich multimodal dataset, called MuVi-Sync, on which we train a novel Affective Multimodal Transformer (AMT) model to generate music given a video. This model includes a novel mechanism to enforce affective similarity between video and music. Finally, post-processing is performed based on a biGRU-based regression model to estimate note density and loudness based on the video features. This ensures a dynamic rendering of the generated chords with varying rhythm and volume. 
In a thorough experiment, we show that our proposed framework can generate music that matches the video content in terms of emotion. The musical quality, along with the quality of music-video matching is confirmed in a user study. The proposed AMT model, along with the new MuVi-Sync dataset, presents a promising step for the new task of music generation for videos._


If you find this resource useful, [please cite the original work](https://arxiv.org/abs/XXX):

      @article{XXX,
        title={Music Generation to Match Video using an Affective Multimodal Transformer model},
        author={Kang, Jaeyong and Poria, Soujanya and Herremans, Dorien},
        journal={arXiv preprint arXiv:XXX},
        year={2023}
      }

  Kang, J., Poria, S. & Herremans, D. (2023). Music Generation to Match Video using an Affective Multimodal Transformer model. arXiv preprint arXiv:XXX.

## Dataset files

## Affective Multimodal Transformer model

## Prerequisites
clip==1.0
coloredlogs==15.0.1
efficientnet_pytorch==0.7.1
ffmpeg_python==0.2.0
ftfy==6.1.1
matplotlib==3.5.3
midi2audio==0.1.1
MIDIUtil==1.2.1
moviepy==1.0.3
music21==7.3.3
numpy==1.19.5
omnizart==0.5.0
opencv_python==4.7.0.72
pandas==1.3.5
Pillow==10.1.0
pretty_midi==0.2.9
pydub==0.25.1
regex==2022.10.31
scenedetect==0.6.1
scikit_learn==1.0.2
scipy==1.7.3
torch==1.12.1
torchvision==0.13.1
