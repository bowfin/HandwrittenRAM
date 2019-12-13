Recurrent Attention Model (RAM)
===============================

This is a RAM implementation based on Google DeepMind research paper "Recurrent Models of Visual Attention". The model is trained on MNIST Dataset.

![Visual Demo](saved/plot-movie.gif)

Requirements
------------

* Python 2.7
* Tensorflow 1.4
* Numpy 1.13
* Imageio 2.3
* Matplotlib 1.4

Implementation
--------------

* Recurrent Neural Network (RNN/LSTM)
* Attention mechanism (RL)
* Action, location and value networks
* Stochastic location and action sampling
* 3-way Glimpse sensor
* MNIST Dataset

Experiments
-----------

A few experiments were conducted to find out how the network can learn with the
attention model from full and partial dataset. The following table illustrates 
how network accuracy and confidence changes with different size of training data.

| Training Data | Test Data     | Accuracy      | Confidence    |
| ------------- | ------------- | ------------- | ------------- |
| 60K           | 10K           | 98 %          | 98 %          |
| 1K            | 10K           | 91 %          | 97 %          |
| 100           | 10K           | 74 %          | 73 %          |

Training
--------

To train the network from scratch call the python script once to save the model:

```bash
python tf-ram.py -t
```

Then call the script again to load and save in a loop (see Issues):

```bash
while (true)
do
  python tf-ram.py -r -t
done
```

Testing
-------

To test the network run python script without any arguments:

```bash
python tf-ram.py
```

Demo
----

The network can run with visual feedback while testing or training. To demo while testing call the following command:

```bash
python tf-ram.py -d -r
```

Issues
------

Partial graph execution in Tensorflow does not seem to close handles, which 
causes excessive memory allocation in the training loop. Due to this issue, the training was done
in an external loop calling python script in train-save-restore fashion.
