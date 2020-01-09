
import h5py
import jetdata
import numpy as np

pulse0 = 80128
pulse1 = 96563

fname = 'tomo_data.hdf'
print('Writing:', fname)
f = h5py.File(fname, 'w')

for pulse in range(pulse0, pulse1+1):
    
    tomo, tomo_t = jetdata.get_tomo(pulse)
    if len(tomo) == 0:
        continue
    print('%-10s %-10s %-20s %-10s' % (pulse, 'tomo', tomo.shape, tomo.dtype))
    print('%-10s %-10s %-20s %-10s' % (pulse, 'tomo_t', tomo_t.shape, tomo_t.dtype))    
    
    bolo, bolo_t = jetdata.get_bolo(pulse)
    if len(bolo) == 0:
        continue
    print('%-10s %-10s %-20s %-10s' % (pulse, 'bolo', bolo.shape, bolo.dtype))
    print('%-10s %-10s %-20s %-10s' % (pulse, 'bolo_t', bolo_t.shape, bolo_t.dtype))    

    bolo, bolo_t = jetdata.resample(bolo, bolo_t, tomo_t)
    print('%-10s %-10s %-20s %-10s' % (pulse, 'bolo', bolo.shape, bolo.dtype))
    print('%-10s %-10s %-20s %-10s' % (pulse, 'bolo_t', bolo_t.shape, bolo_t.dtype))    

    g = f.create_group(str(pulse))
    g.create_dataset('tomo', data=tomo)
    g.create_dataset('tomo_t', data=tomo_t)
    g.create_dataset('bolo', data=bolo)
    g.create_dataset('bolo_t', data=bolo_t)
    print('-'*50)

f.close()
