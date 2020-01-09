
import h5py
import jetdata
import numpy as np

# ----------------------------------------------------------------------

pulses = [94988,
          95128,
          95131,
          95132,
          95135,
          95136,
          95137,
          95174,
          95725,
          95727,
          95729,
          95733,
          96254,
          96256]

print('pulses:', pulses)

# ----------------------------------------------------------------------

fname = 'bolo_data.hdf'
print('Writing:', fname)
f = h5py.File(fname, 'a')

# ----------------------------------------------------------------------

for pulse in pulses:

    bolo, bolo_t = jetdata.get_bolo(pulse)
    print('%-10s %-10s %-20s %-10s' % (pulse, 'bolo', bolo.shape, bolo.dtype))
    print('%-10s %-10s %-20s %-10s' % (pulse, 'bolo_t', bolo_t.shape, bolo_t.dtype))

    pulse = str(pulse)
    if pulse in f:
        del f[pulse]

    g = f.create_group(pulse)
    g.create_dataset('bolo', data=bolo)
    g.create_dataset('bolo_t', data=bolo_t)
    print('-'*50)

# ----------------------------------------------------------------------

f.close()
