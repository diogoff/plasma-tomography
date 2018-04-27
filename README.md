# A Deep Neural Network for Plasma Tomography


### Requirements

- Keras, TensorFlow


### Instructions

1. Run `get_data.py` to get all of the available tomographic reconstructions from bolometer data.

    - This script will only work from within one of the JET computing clusters (JAC or Freia).

    - It will produce an output file called `tomo_data.hdf`.

2. (Optional) Run `read_data.py` to check that the file `tomo_data.hdf` is readable.

3. (Optional) Run `plot_data.py` to plot some sample reconstructions.

    - Sample reconstructions will be selected randomly. Hit `Ctrl-C` to finish.

4. Run `input_data.py` to create the arrays that will be used as input to train the network.

    - The script will use `tomo_data.hdf` as input.

    - It will create two output files: `X_train.npy` and `Y_train.npy`.

5. Run `model_train.py` to train the network.

    - Before running this script, set the `gpus` parameter in the call to `multi_gpu_model()` and set the `batch_size` to be used in `parallel_model.fit()`.
    
        - `gpus` should be set to the number of available GPUs.
        
        - `batch_size` should be an exact multiple of `gpus` and an approximate divisor of the number of training samples. Pick the largest such value within the constraints of available GPU memory.

    - Training will finish automatically once there is no hope of improving the validation loss further.
    
    - The model parameters corresponding to the minimum validation loss will be saved in `model_weights.hdf`.

6. (Optional) During training, run `plot_train.py` to see how loss and validation loss are evolving.

    - The plot will also indicate the epoch where the minimum validation loss was achieved.
    
7. (Optional) After training, run `model_validate.py` to test the model on the validation data.

    - This script does not need to be run on the GPU, it can run with TensorFlow on the CPU.
    
    - It needs `X_train.npy` and `Y_train.npy` in order to extract the validation data.
    
    - After running this script, check that `val_loss` is the same as reported by `plot_train.py` (apart from some rounding error).


### References

- [Full-Pulse Tomographic Reconstruction with Deep Neural Networks](https://arxiv.org/pdf/1802.02242.pdf) - D. R. Ferreira, P. J. Carvalho, H. Fernandes (2018)

- [Deep learning for plasma tomography using the bolometer system at JET](https://arxiv.org/pdf/1701.00322.pdf) - F. A. Matos, D. R. Ferreira, P. J. Carvalho (2017)
