# Deep Neural Network for Plasma Tomography

### Instructions

- Run `python get_tomo.py` to get all the available tomographic reconstructions from bolometer data.

    - This will only work from within the JET computing clusters (JAC or Freia).

    - It will produce an output file called `tomo_data.hdf`.

- (Optional) Run `python read_data.py` to check that the file `tomo_data.hdf` is readable.

- (Optional) Run `python plot_data.py` to plot some sample reconstructions.

    - Sample reconstructions will be selected randomly.
    
    - Hit Ctrl-C to finish.




### References

- [Full-Pulse Tomographic Reconstruction with Deep Neural Networks](https://arxiv.org/pdf/1802.02242.pdf) - D. R. Ferreira, P. J. Carvalho, H. Fernandes (2018)

- [Deep learning for plasma tomography using the bolometer system at JET](https://arxiv.org/pdf/1701.00322.pdf) - F. A. Matos, D. R. Ferreira, P. J. Carvalho (2017)
