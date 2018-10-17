# Sequence-to-sequence Domain adaptation \\for robust text image recognition
![model](https://github.com/Seq2seqAdapt/Seq2seqAdapt_paper/blob/master/model_framework.jpg)
# Prerequsites
tensorflow-gpu == 1.10.0
python 3.6.3
python
Besides, we use python package distance to calculate edit distance for evaluation.

- Tensorflow: [Installation Instructions](https://www.tensorflow.org/get_started/os_setup#download-and-setup) 
- Distance (Optional):

```
wget http://www.cs.cmu.edu/~yuntiand/Distance-0.1.3.tar.gz
```

```
tar zxf Distance-0.1.3.tar.gz
```

```
cd distance; sudo python setup.py install
```

# Dataset

For a toy sample, we can download the following datasets.

## Source synthetic text images

   - A subset of [Synth 90k](http://www.robots.ox.ac.uk/~vgg/data/text/):
```
wget http://www.cs.cmu.edu/~yuntiand/sample.tgz
```

```
tar zxf sample.tgz
```
## Target text images

  -  Real scene text images from ICDAR03, ICDAR13, IIIT5k and SVT:

```
wget http://www.cs.cmu.edu/~yuntiand/evaluation_data.tgz
```

```
tar zxf evaluation_data.tgz
```

# Usage:

(0) Preparing dataset

-Suppose DATA_HOME=/home/data/OCR

 ```
   python gen_tfrecord.py
 ```
(1) Pretraining a source model

```
   python main_baseline.py --phase='train'
```
(2) Training a domain adaptation model

```
   python main.py --phase='train'
```

# Parameters:

#### Parameters for dataset
```
vi defaults_dataset.py
```
 * `DATA_HOME`:  The base directory of the dataset, default is '/home/data/OCR/'
 * `DATASET `:   the name of dataset, eg. svt, sample

#### Parameters for training
```
vi defaults.py
```
 * `DATA_HOME`:  The base directory of the dataset, default is '/home/data/OCR/'
 * `SOURCE_DATASET `:   the name of source dataset, eg. sample
 * `TARGET_DATASET `:   the name of target dataset, eg. svt

 * `MODEL_DIR `:   the directory of model
 * `OUTPUT_DIR `:   output directory
 * `LOG_PATH `:   the logging file

