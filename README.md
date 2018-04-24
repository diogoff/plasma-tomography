# A Deep Neural Network for Plasma Tomography


### Requirements

- Keras

- TensorFlow


### Instructions

- Run `python get_data.py` to get all the available tomographic reconstructions from bolometer data.

    - This will only work from within the JET computing clusters (JAC or Freia).

    - It will produce an output file called `tomo_data.hdf`.

- (Optional) Run `python read_data.py` to check that the file `tomo_data.hdf` is readable.

- (Optional) Run `python plot_data.py` to plot some sample reconstructions.

    - Sample reconstructions will be selected randomly.
    
    - Hit `Ctrl-C` to finish.

- Run `python input_data.py` to create the arrays that will be used as input to train the network.

    - This script will use the file `tomo_data.hdf` as input.

    - It will create two output files: `X_train.npy` and `Y_train.npy`.

- Run `TF_CPP_MIN_LOG_LEVEL=3 python -W ignore model_train.py` to train the network.

    - Before running this command, set the `gpus` parameter in `multi_gpu_model()` and set the `batch_size` to be used in `parallel_model.fit()`.
    
        - `gpus` should be the number of GPUs available in your machine.
        
        - `batch_size` should be a multiple of `gpus` and a divisor of the number of samples used for training, while at the same time being bound by GPU memory available.

    - Training will finish automatically once there is no hope of further improving the validation loss.
    
    - The model parameters corresponding to the minimum validation loss will be saved in `model_weights.hdf`.

- (Optional) During training, run `python plot_train.py` to see the evolution of loss and validation loss, as well as the epoch where the minimum validation loss was achieved.


### References

- [Full-Pulse Tomographic Reconstruction with Deep Neural Networks](https://arxiv.org/pdf/1802.02242.pdf) - D. R. Ferreira, P. J. Carvalho, H. Fernandes (2018)

- [Deep learning for plasma tomography using the bolometer system at JET](https://arxiv.org/pdf/1701.00322.pdf) - F. A. Matos, D. R. Ferreira, P. J. Carvalho (2017)
