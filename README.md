# Realistic Fingerprint Generator 

Tensorflow implementation of realistic fingerprint generator based on DCGAN [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434).

![result](assets/fingerprint_traning.gif)


## Prerequisites

- Python 3.6+
- [Tensorflow 1.12.0](https://github.com/tensorflow/tensorflow/tree/r1.12)
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)

#### For training:
- Download NIST Special Database 14.
- Move to "./data" folder

#### For testing:
- Download [Pre trained model](https://drive.google.com/drive/folders/1Ru6YL_5Wvo1sGR-YK-c0FNlLtbln19Vb?usp=sharing)
- Extract checkpoint folder to the main root (under "-Realistic-Fingerprint-Generator").

## Usage

Generate fingerprint with pre-trained model (NIST14):

    $ python main.py --input_height=650 --output_height=650 --test
    
To train a model with NIST14 dataset:

    $ python main.py --dataset nist14 --input_height=650 --output_height=650 --train --crop=True

Or, you can use your own dataset (without central crop) by:

    $ mkdir data/DATASET_NAME
    ... add images to data/DATASET_NAME ...
    $ python main.py --dataset DATASET_NAME --train
    $ python main.py --dataset DATASET_NAME
    $ # example
    $ python main.py --dataset=Fimages --input_fname_pattern="*.tif" --train

## Evaluation 
Use state-of-the-art fingerprint feature extractor [fingerNet](https://arxiv.org/abs/1709.02228) to extract some statistics from real and generated images

|Data set | Mean number of minute | STD number of minute | Mean orientation of minute | STD orientation of minute |
--- | --- | --- | --- |--- |
|Nist14 | 123.39 | 30.63 | 3.30 | 1.85 |
|Generated | 123.87 | 30.53 | 3.29 | 1.88 |

## Related works

- [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
- [fingerNet](https://arxiv.org/abs/1709.02228)


## Author

Rafael Bouzaglo / [@rafaelbou]
