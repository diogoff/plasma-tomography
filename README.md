## A Deconvolutional Neural Network for Plasma Tomography

This repository contains a neural network that produces tomographic reconstructions similar to those available at JET.

### Instructions

- Run `train_data.py` to get all the available tomographic reconstructions from bolometer data.

    - This script will only run on a JET computing cluster (e.g. Freia).

    - An output file `train_data.hdf` will be created.

- Run `split_data.py` to split the train data into training set and validation set.

    - This will create `X_train.npy`, `Y_train.npy`, `X_valid.npy`, and `Y_valid.npy`.

- Run `model_train.py` to train the model.

    - Training will finish automatically once the validation loss no longer improves.
    
    - The best model parameters will be saved in `model_weights.hdf`.

- During training, run `plot_loss.py` to see how the loss and validation loss are evolving.

    - The script will indicate the epoch where the minimum validation loss was achieved.
    
- After training, run `model_valid.py` to test the model on the validation data.

    - Check that the reported `val_loss` is the same as indicated by `plot_loss.py`.

- Run `model_test.py` to test the model on a given pulse.

    - Before running this script, set the desired pulse, start time (`t0`), end time (`t1`), and time step (`dt`).

    - Since this script will grab the bolometer data for the test pulse, it needs to run on a JET computing cluster.
    
    - An output file `test_data.hdf` will be created.

- Run `plot_frames.py` to plot the reconstructions generated by the model for the test pulse.

    - If needed, adjust `vmax` to change the dynamic range of the plots.

- Run `create_movie.py` to produce a movie of the reconstructions for the test pulse.

    - If needed, adjust `vmax` to change the dynamic range.


### References

- [Full-Pulse Tomographic Reconstruction with Deep Neural Networks](https://arxiv.org/pdf/1802.02242.pdf) - D. R. Ferreira, P. J. Carvalho, H. Fernandes (2018)

- [Deep learning for plasma tomography using the bolometer system at JET](https://arxiv.org/pdf/1701.00322.pdf) - F. A. Matos, D. R. Ferreira, P. J. Carvalho (2017)
