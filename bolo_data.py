
import sys
import h5py
import jetdata
import numpy as np

# ----------------------------------------------------------------------

if len(sys.argv) < 2:
    print('Usage: %s pulse pulse pulse ...' % sys.argv[0])
    exit()
    
pulses = [int(pulse) for pulse in sys.argv[1:]]
print('pulses:', pulses)

# ----------------------------------------------------------------------

fname = 'bolo_data.hdf'
print('Writing:', fname)
f = h5py.File(fname, 'a')

# ----------------------------------------------------------------------

for pulse in pulses:

    bolo, bolo_t = jetdata.get_bolo(pulse)
    if len(bolo) == 0:
        continue
    print('%-10s %-10s %-20s %-10s' % (pulse, 'bolo', bolo.shape, bolo.dtype))
    print('%-10s %-10s %-20s %-10s' % (pulse, 'bolo_t', bolo_t.shape, bolo_t.dtype))

    dt = 0.005
    new_t = np.arange(40., bolo_t[-1]-dt, dt)
    bolo, bolo_t = jetdata.resample(bolo, bolo_t, new_t)
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
