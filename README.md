## A Deconvolutional Neural Network for Plasma Tomography

This repository contains a neural network that produces tomographic reconstructions similar to those available at JET.

### Requirements

- Python 2.7, Keras 2.2.4, TensorFlow 1.13.1

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

1. Run `python tomo_data.py` to get all the available tomographic reconstructions for training.

    - This script will only run on the JET computing cluster (e.g. Freia).
    
    - An output file `tomo_data.hdf` will be created.

2. Run `python split_data.py` to split the data into training/validation/test sets.

    - This will create three datasets: (`X_train.npy`, `Y_train.npy`), (`X_valid.npy`, `Y_valid.npy`) and (`X_test.npy`, `Y_test.npy`).

3. Run `python model_train.py` to train the model.

    - Training will finish automatically once the validation loss no longer improves.
    
    - The best model will be saved in `model.hdf`.

4. After (or during) training, run `python plot_train.py` to plot the loss and validation loss across epochs.

    - The script will also indicate the epoch where the minimum validation loss was achieved.
    
5. After training, run `python model_valid.py` to test the model on the validation and test sets.

    - Check that the reported `loss` for the validation set is the same as indicated by `plot_train.py`.

6. Run `python bolo_data.py 92213` to get the bolometer data for a test pulse.

    - Since this script will grab the bolometer data for the test pulse, it needs to run on the JET computing cluster.

    - The data will be appended to `bolo_data.hdf`. This file will be created, if it does not exist.
    
7. Run `python model_predict.py` to generate the reconstructions for the pulses in `bolo_data.hdf`.

    - The results will be appended to each test pulse in `bolo_data.hdf`.

8. Run `python plot_frames.py 92213 46.40 54.70 0.01` to plot the reconstructions from `bolo_data.hdf`.

    - The command-line arguments specify the pulse, start time (`t0`), end time (`t1`) and time step (`dt`) for the plots.

    - If needed, adjust `vmax` to change the dynamic range of the plots (in MW/m3).

9. Run `python plot_movie.py 92213 46.40 54.70 0.01` to produce a movie of the reconstructions for a pulse.

    - The command-line arguments specify the pulse, start time (`t0`), end time (`t1`) and time step (`dt`) for the movie.

    - If needed, adjust `vmax` to change the dynamic range of the plots (in MW/m3).

    - If needed, adjust `fps` to change the frame rate.

### References

- D.R. Ferreira, P.J. Carvalho, H. Fernandes, [Deep Learning for Plasma Tomography and Disruption Prediction from Bolometer Data](https://arxiv.org/pdf/1910.13257.pdf), IEEE Transactions on Plasma Science, 2019 [[to appear](https://ieeexplore.ieee.org/document/8882311)]

- D.R. Ferreira, P.J. Carvalho, H. Fernandes, [Full-Pulse Tomographic Reconstruction with Deep Neural Networks](https://arxiv.org/pdf/1802.02242.pdf), Fusion Science and Technology, vol. 74, no. 1-2, pp. 47-56, 2018 [[BibTeX](https://www.tandfonline.com/action/downloadCitation?doi=10.1080/15361055.2017.1390386&format=bibtex)]

- F.A. Matos, D.R. Ferreira, P.J. Carvalho, [Deep learning for plasma tomography using the bolometer system at JET](https://arxiv.org/pdf/1701.00322.pdf), Fusion Engineering and Design, vol. 114, pp. 18-25, Jan. 2017 [[BibTeX](https://www.sciencedirect.com/sdfe/arp/cite?pii=S0920379616306883&format=text%2Fx-bibtex&withabstract=false)]
