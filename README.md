# Bahdanau Attention Implementation in Keras
I have implemented a Bahdanau style Attention layer in Keras. I tried to keep things as simple as possible. Without the beautiful implementation as in [Datalogue](https://github.com/datalogue/keras-attention), a fantastic tutorial in [this link](https://medium.com/datalogue/attention-in-keras-1892773a4f22) by [Zafarali Ahmed](https://medium.com/@zafarali) it would not be possible. I am also indebted to this nice [tutorial](https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/). Finally withouth this [paper](https://arxiv.org/abs/1409.0473) all of these would be a staff of dreams.
## Features
* A one layer Sequence to Sequence model with custom LSTM cell for Attention mechanism
* Support for different input and output length sequence
* A simple data generator module to generate some training sequence
* A module to visulaize the effect of attention weights for input and output sequence
## Requirement
Standard Python 2.7 installation with Keras 2, numpy, matplotlib
## Usage
1. First run `Datagenerator.py`. This will automatically create a file `time_corpus`. Each row of this file has two strings, the first one (X) is network input with three elements weekday(w)[0-7], time(t)[0-23] and month(m)[0-11]. The second one (Y) is the word corresponding to weekday, time and month, also number of days in that month is appended. So, length of X is 3 and length of Y is 4. Note that m, t and w can appear in any order. 
