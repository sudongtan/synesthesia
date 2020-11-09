# Synesthesia

Code and media samples for the paper [Automated Music Generation for Visual Art through Emotion](http://computationalcreativity.net/iccc20/papers/137-iccc20.pdf).

## Generated Samples
||||||
|:----:|:----:|:----:|:----:|:----:|
|Input|![6](/dataset/image/test/6.jpg)|![12](/dataset/image/test/12.jpg)|![10](/dataset/image/test/10.jpg)|![8](/dataset/image/test/8.jpg)|
|Output|[midi](/output/samples/09.mid)|[midi](/output/samples/03.mid)|[midi](/output/samples/05.mid)|[midi](/output/samples/07.mid)|
|Input|![14](/dataset/image/test/14.jpg)|![4](/dataset/image/test/4.jpg)|![2](/dataset/image/test/2.jpg)|![1](/dataset/image/test/1.jpg)|
|Output|[midi](/output/samples/01.mid)|[midi](/output/samples/11.mid)|[midi](/output/samples/13.mid)|[midi](/output/samples/14.mid)|
|Input|![13](/dataset/image/test/13.jpg)|![11](/dataset/image/test/11.jpg)|![9](/dataset/image/test/9.jpg)|![5](/dataset/image/test/5.jpg)|
|Output|[midi](/output/samples/02.mid)|[midi](/output/samples/04.mid)|[midi](/output/samples/06.mid)|[midi](/output/samples/10.mid)|
|Input||![3](/dataset/image/test/3.jpg)|![7](/dataset/image/test/7.jpg)||
|Output||[midi](/output/samples/12.mid)|[midi](/output/samples/08.mid)||

Sources of the images: https://www.imageemotion.org/

## Instructions

### Dependencies

- `python3.7`
- `cuda10`
- `musescore`

The remaining python dependencies can be installed with:

    pip install -r requirements.txt

### Training dataset

The full dataset can be downloaded from https://www.cs.rochester.edu/u/qyou/deepemotion/ .

### PerformanceRNN

```bash
cd rnn
python preprocess.py ../dataset/midi/train ../dataset/midi/rnn # preprocess data
python train.py -s ../model/rnn_example.sess -d ../dataset/midi/rnn -i 10 # train model
python generate.py -i ../dataset/image/test/1.jpg #generate music from a given image
```
The implementation of performance RNN is modified from https://github.com/djosix/Performance-RNN-PyTorch .


### Transformer

```bash
cd transformer
python preprocess.py # preprocess data
python train.py # train model
python generate.py -i ../dataset/image/test/1.jpg #generate music from a given image
```

The implementation of transformer is modified from https://github.com/bearpelican/musicautobot .
