
import ppf
import h5py
import numpy as np

ppf.ppfssr([0,1,2,3,4])

def get_tomo(pulse):
    ppf.ppfgo(pulse)
    tomo = []
    tomo_t = []
    for uid in ['jetppf', 'bolom']:
        ppf.ppfuid(uid, 'r')
        for i in range(1, 100):
            pr = 'pr%02d' % i
            ihdata, iwdata, data, x, t, ier = ppf.ppfget(pulse, 'bolt', pr, reshape=1)
            if ier != 0:
                break
            t = np.float32(ihdata.strip().split()[-1][2:-1])
            data = np.flipud(np.transpose(data))
            tomo.append(data)
            tomo_t.append(t)
    if len(tomo) > 0:
        tomo = np.array(tomo)
        tomo_t = np.array(tomo_t)
        ix = np.argsort(tomo_t)
        tomo = tomo[ix]
        tomo_t = tomo_t[ix]
    return tomo, tomo_t

def get_bolo(pulse):
    ppf.ppfgo(pulse)
    ppf.ppfuid('jetppf', 'r')
    ihdata, iwdata, kb5h, x, kb5h_t, ier = ppf.ppfget(pulse, 'bolo', 'kb5h', reshape=1)
    ihdata, iwdata, kb5v, x, kb5v_t, ier = ppf.ppfget(pulse, 'bolo', 'kb5v', reshape=1)
    bolo = np.hstack((kb5h, kb5v))
    assert np.all(kb5h_t == kb5v_t)
    bolo_t = kb5h_t
    return bolo, bolo_t

pulse0 = 80128
pulse1 = 96563

fname = 'tomo_data.hdf'
print('Writing:', fname)
f = h5py.File(fname, 'w')

for pulse in range(pulse0, pulse1+1):
    
    tomo, tomo_t = get_tomo(pulse)
    if len(tomo) == 0:
        continue
    print('%-10s %-10s %-20s %-10s' % (pulse, 'tomo', tomo.shape, tomo.dtype))
    print('%-10s %-10s %-20s %-10s' % (pulse, 'tomo_t', tomo_t.shape, tomo_t.dtype))    
    
    bolo, bolo_t = get_bolo(pulse)
    if len(bolo) == 0:
        continue
    print('%-10s %-10s %-20s %-10s' % (pulse, 'bolo', bolo.shape, bolo.dtype))
    print('%-10s %-10s %-20s %-10s' % (pulse, 'bolo_t', bolo_t.shape, bolo_t.dtype))    

    g = f.create_group(str(pulse))
    g.create_dataset('tomo', data=tomo)
    g.create_dataset('tomo_t', data=tomo_t)
    g.create_dataset('bolo', data=bolo)
    g.create_dataset('bolo_t', data=bolo_t)
    print('-'*50)

f.close()
