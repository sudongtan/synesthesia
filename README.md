# Synesthesia

## TODO
data clean up
 - add sample image, sample midi, add reference, 
 - sample model
 - save final model and 
 - clean up evaluation folder
 - clean up google folder
  
display generated midi

code clean up
 - add more keywords args
 - remove unecessary prints
 - clean code not in use

Test run on different computers
  
## Overview

## Generated Samples


## Instructions

##

## Installation
- `python3.7`
- `cuda10`
- `musescore`
- `python dependencies`
```bash
pip install -r requirements.txt
```

### PerformanceRNN

- to generate music from a given image using pre-trained model

```bash
cd rnn
python generate.py -i ../dataset/image/test/1.jpg
```
- to train new model
```bash
cd rnn
python preprocess.py ../dataset/midi/train ../dataset/midi/rnn
python train.py -s ../model/rnn_example.sess -d ../dataset/midi/rnn -i 10
```
The implementation of performance RNNS modified from https://github.com/djosix/Performance-RNN-PyTorch

### Transformer
- to generate music from a given image using pre-trained model

```bash
cd transformer
python generate.py -i ../dataset/image/test/1.jpg
```
- to train new model
```bash
cd transformer
python preprocess.py
python train.py
```

- the implementation of transformer is modified from https://github.com/bearpelican/musicautobot