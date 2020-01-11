## A Deconvolutional Neural Network for Plasma Tomography

This repository contains a neural network that produces tomographic reconstructions similar to those available at JET.

### Requirements

- Python 3, TensorFlow 2.1, CUDA 10.1, cuDNN 7.6

### Instructions

1. Run `python3 tomo_data.py` to get all the available tomographic reconstructions and the corresponding bolometer signals.

    - This script will only run on a JET computing cluster (e.g. Freia).
    
    - An output file `tomo_data.h5` will be created.

2. Run `python3 split_data.py` to split the data into training set and validation set.

    - This will create two datasets: (`X_train.npy`, `Y_train.npy`) and (`X_valid.npy`, `Y_valid.npy`).

3. Run `python3 batch_size.py` to determine the batch size that should be used during training.

    - Adjust `n_gpus` to reflect the number of GPUs to be used during training.

4. Run `python3 model_train.py` to train the model.

    - Adjust `batch_size` according to the result of the previous script.

    - Training will finish automatically once the validation loss no longer improves.
    
    - The model will be saved in `model.h5`.

5. After (or during) training, run `python3 plot_train.py` to plot the loss and validation loss across epochs.

    - The script will also indicate the epoch where the minimum validation loss was achieved.
    
6. After training, run `python3 model_valid.py` to test the model on the validation set.

    - Check that the reported `loss` for the validation set is the same as indicated by `plot_train.py`.

7. Run `python3 plot_frames.py 92213 48.0 54.0 0.01 1.0` to plot the reconstructions for a test pulse.

    - The command-line arguments specify the pulse, start time (`t0`), end time (`t1`), time step (`dt`) and dynamic range (`vmax` in MW/m3) for the plots.

8. Run `python3 plot_movie.py 92213 48.0 54.0 0.01 1.0` to produce a movie of the reconstructions for a test pulse.

    - The command-line arguments specify the pulse, start time (`t0`), end time (`t1`), time step (`dt`) and dynamic range (`vmax` in MW/m3) for the movie.

    - If needed, adjust `fps` to change the frame rate.

### References

- D.R. Ferreira, P.J. Carvalho, H. Fernandes, [Deep Learning for Plasma Tomography and Disruption Prediction from Bolometer Data](https://arxiv.org/pdf/1910.13257.pdf), IEEE Transactions on Plasma Science, 2019 ([to appear](https://ieeexplore.ieee.org/document/8882311))

- D.R. Ferreira, P.J. Carvalho, H. Fernandes, [Full-Pulse Tomographic Reconstruction with Deep Neural Networks](https://arxiv.org/pdf/1802.02242.pdf), Fusion Science and Technology, vol. 74, no. 1-2, pp. 47-56, 2018 [[BibTeX](https://www.tandfonline.com/action/downloadCitation?doi=10.1080/15361055.2017.1390386&format=bibtex)]

- F.A. Matos, D.R. Ferreira, P.J. Carvalho, [Deep learning for plasma tomography using the bolometer system at JET](https://arxiv.org/pdf/1701.00322.pdf), Fusion Engineering and Design, vol. 114, pp. 18-25, Jan. 2017 [[BibTeX](https://www.sciencedirect.com/sdfe/arp/cite?pii=S0920379616306883&format=text%2Fx-bibtex&withabstract=false)]
