
import h5py
import jetdata
import numpy as np

# ------------------------------------------------------------------------------

def resample(bolo, bolo_t, new_t):
    dt = 0.005
    new = []
    for t in new_t:
        i0 = np.argmin(np.fabs(bolo_t - t))
        i1 = np.argmin(np.fabs(bolo_t - (t + dt)))
        new.append(np.mean(bolo[i0:i1+1], axis=0))
    new = np.array(new)
    return new, new_t

# ------------------------------------------------------------------------------

pulse0 = 80128
pulse1 = 96563

fname = 'tomo_data.h5'
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

    bolo, bolo_t = resample(bolo, bolo_t, tomo_t)
    print('%-10s %-10s %-20s %-10s' % (pulse, 'bolo', bolo.shape, bolo.dtype))
    print('%-10s %-10s %-20s %-10s' % (pulse, 'bolo_t', bolo_t.shape, bolo_t.dtype))    

    g = f.create_group(str(pulse))
    g.create_dataset('tomo', data=tomo)
    g.create_dataset('tomo_t', data=tomo_t)
    g.create_dataset('bolo', data=bolo)
    g.create_dataset('bolo_t', data=bolo_t)
    print('-'*50)

f.close()
