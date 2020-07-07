

 TensorflowTTS provides real-time state-of-the-art speech synthesis architectures such as Tacotron-2, Melgan, Multiband-Melgan, FastSpeech, FastSpeech2 based-on TensorFlow 2. With Tensorflow 2, we can speed-up training/inference progress, optimizer further by using [fake-quantize aware](https://www.tensorflow.org/model_optimization/guide/quantization/training_comprehensive_guide) and [pruning](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras), make TTS models can be run faster than real-time and be able to deploy on mobile devices or embedded systems. 

## Features
- High performance on Speech Synthesis.
- Be able to fine-tune on other languages.
- Fast, Scalable and Reliable.
- Suitable for deployment.
- Easy to implement new model based-on abtract class.
- Mixed precision to speed-up training if posible.

## Requirements
This repository is tested on Ubuntu 18.04 with:

- Python 3.6+
- Cuda 10.1
- CuDNN 7.6.5
- Tensorflow 2.2
- [Tensorflow Addons](https://github.com/tensorflow/addons) 0.9.1

Different Tensorflow version should be working but not tested yet. This repo will try to work with latest stable tensorflow version.

