## A Deconvolutional Neural Network for Plasma Tomography

This repository contains a neural network that produces tomographic reconstructions similar to those available at JET.

### Requirements

- Keras 2.1.2, TensorFlow 1.4.1

- Configure `~/.keras/keras.json` as follows:

```
{
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
    "image_data_format": "channels_last",
}

```

### Instructions

1. Run `train_data.py` to get all the available tomographic reconstructions for training.

    - This script will only run on the JET computing cluster (e.g. Freia).
    
    - An output file `train_data.hdf` will be created.

2. Run `train_split.py` to split the train data into training set and validation set.

    - This will create two datasets: (`X_train.npy`, `Y_train.npy`) and (`X_valid.npy`, `Y_valid.npy`).

3. Run `train_model.py` to train the model.

    - Training will finish automatically once the validation loss no longer improves.
    
    - The best model will be saved in `model.hdf`.

4. After (or during) training, run `train_plot.py` to plot the loss and validation loss across epochs.

    - The script will also indicate the epoch where the minimum validation loss was achieved.
    
5. After training, run `test_valid.py` to test the model on the validation data.

    - Check that the reported `val_loss` is the same as indicated by `train_plot.py`.

6. Run `test_data.py 92213` to get the bolometer data for a test pulse.

    - Since this script will grab the bolometer data for the test pulse, it needs to run on the JET computing cluster.

    - The data will be appended to `test_data.hdf`. This file will be created, if it does not exist.
    
7. Run `test_model.py` to generate the reconstructions for the test pulse(s).

    - The results will be appended to each test pulse in `test_data.hdf`.

8. Run `test_frames.py 92213 46.40 54.70 0.01` to plot the reconstructions for a test pulse.

    - The command-line arguments specify the pulse, start time (`t0`), end time (`t1`) and time step (`dt`) for the plots.

    - If needed, adjust `vmax` to change the dynamic range of the plots (in MW/m3).

9. Run `test_movie.py 92213 46.40 54.70 0.01` to produce a movie of the reconstructions for a pulse.

    - The command-line arguments specify the pulse, start time (`t0`), end time (`t1`) and time step (`dt`) for the movie.

    - If needed, adjust `vmax` to change the dynamic range of the plots (in MW/m3).

    - If needed, adjust `fps` to change the frame rate.

### Reference

- D.R. Ferreira, P.J. Carvalho, H. Fernandes, [Full-Pulse Tomographic Reconstruction with Deep Neural Networks](https://arxiv.org/pdf/1802.02242.pdf), Fusion Science and Technology, vol. 74, no. 1-2, pp. 47-56, 2018 [[BibTeX](https://www.tandfonline.com/action/downloadCitation?doi=10.1080/15361055.2017.1390386&format=bibtex)]
