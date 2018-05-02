# A Deep Neural Network for Plasma Tomography


### Requirements

- Keras, TensorFlow


### Instructions

- Run `train_data.py` to get all the available tomographic reconstructions from bolometer data.

    - This script will only run on a JET computing cluster (e.g. Freia).

    - An output file `train_data.hdf` will be created.

- Run `split_data.py` to split the train data into training set and validation set.

    - This will create `X_train.npy`, `Y_train.npy`, `X_valid.npy`, and `Y_valid.npy`.

- Run `model_train.py` to train the model.

    - Before running this script, set the number of GPUs and the batch size accordingly.

        - In the call to `multi_gpu_model()`, the `gpus` parameter should be set to the number of available GPUs.
        
        - In `parallel_model.fit()`, `batch_size` should be an exact multiple of `gpus` and an approximate divisor of the number of training samples. Pick the largest value within the constraints of the available GPU memory.

    - Training will finish automatically once the validation loss no longer improves.
    
    - The best model parameters will be saved in `model_weights.hdf`.

- During training, run `plot_loss.py` to see how the loss and validation loss are evolving.

    - The script will indicate the epoch where the minimum validation loss was achieved.
    
- After training, run `model_validate.py` to test the model on the validation data.

    - This script does not need to run on the GPU, it can run with TensorFlow on the CPU.
    
    - Check that the reported `val_loss` is the same as indicated by `plot_loss.py`.

- Run `model_test.py` to test the model on a given pulse.

    - Before running this script, set the desired pulse, start time, and time step.

    - Since this script will grab the bolometer data for the test pulse, it needs to run on a JET computing cluster.
    
    - Also, Tensorflow and Keras need to be installed for this script to run.
    
    - An output file `test_data.hdf` will be created.

- Run `plot_frames.py` to plot the reconstructions generated by the model for the test pulse.

    - If needed, adjust `vmax` to change the dynamic range of the plots.

- Run `plot_movie.py` to create a movie of the reconstructions for the test pulse.

    - If needed, adjust `vmax` to change the dynamic range.


### References

- [Full-Pulse Tomographic Reconstruction with Deep Neural Networks](https://arxiv.org/pdf/1802.02242.pdf) - D. R. Ferreira, P. J. Carvalho, H. Fernandes (2018)

- [Deep learning for plasma tomography using the bolometer system at JET](https://arxiv.org/pdf/1701.00322.pdf) - F. A. Matos, D. R. Ferreira, P. J. Carvalho (2017)
