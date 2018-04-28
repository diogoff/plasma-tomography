# A Deep Neural Network for Plasma Tomography


### Requirements

- Keras, TensorFlow


### Instructions

- Run `train_data.py` to get all the available tomographic reconstructions from bolometer data.

    - This script will only run on a JET computing cluster (JAC or Freia).

    - An output file `train_data.hdf` will be created.

- Run `split_data.py` to split the train data into training set and validation set.

    - This will create `X_train.npy`, `Y_train.npy`, `X_valid.npy`, and `Y_valid.npy`.

- Run `model_train.py` to train the model.

    - Before running this script, set the number of GPUs and the batch size accordingly.

    - Training will finish automatically once the validation loss no longer improves.
    
    - The best model parameters will be saved in `model_weights.hdf`.

- During training, run `plot_train.py` to see how the loss and validation loss are evolving.

    - The script will indicate the epoch where the minimum validation loss was achieved.
    
- After training, run `model_validate.py` to test the model on the validation data.

    - This script does not need to be run on the GPU, it can run with TensorFlow on the CPU.
    
    - Check that the reported `val_loss` is the same as indicated by `plot_train.py`.


### References

- [Full-Pulse Tomographic Reconstruction with Deep Neural Networks](https://arxiv.org/pdf/1802.02242.pdf) - D. R. Ferreira, P. J. Carvalho, H. Fernandes (2018)

- [Deep learning for plasma tomography using the bolometer system at JET](https://arxiv.org/pdf/1701.00322.pdf) - F. A. Matos, D. R. Ferreira, P. J. Carvalho (2017)
