from __future__ import print_function

import numpy as np
from ppf import *

ppfgo()
ppfssr(i=[0,1,2,3,4])

def get_tomo(pulse):
    tomo = []
    tomo_t = []
    for uid in ['jetppf', 'bolom']:
        ppfuid(uid, 'r')
        for i in range(1, 100):
            pr = 'pr%02d' % i
            ihdata, iwdata, data, x, t, ier = ppfget(pulse, 'bolt', pr, reshape=1)
            if ier != 0:
                break
            t = np.float32(ihdata.strip().split()[-1][2:-1])
            tomo.append(np.flipud(np.transpose(data)))
            tomo_t.append(t)
    if len(tomo) > 0:
        tomo = np.array(tomo)
        tomo_t = np.array(tomo_t)
        inds = np.argsort(tomo_t)
        tomo = tomo[inds]
        tomo_t = tomo_t[inds]
        print(pulse, 'tomo:', tomo.shape, tomo.dtype)
        print(pulse, 'tomo_t:', tomo_t.shape, tomo_t.dtype)
    return tomo, tomo_t

def get_bolo(pulse, bolo_t):
    dt = 0.0025
    ppfuid('jetppf', 'r')
    ihdata, iwdata, kb5h, x, kb5h_t, ier = ppfget(pulse, 'bolo', 'kb5h', reshape=1)
    ihdata, iwdata, kb5v, x, kb5v_t, ier = ppfget(pulse, 'bolo', 'kb5v', reshape=1)
    assert np.all(kb5h_t == kb5v_t)
    kb5 = np.hstack((kb5h, kb5v))
    kb5_t = kb5h_t
    t0 = kb5_t[0]
    t1 = kb5_t[-1]-2.*dt
    if bolo_t[0] < t0:
        i = np.argmin(np.fabs(bolo_t - t0))
        bolo_t = bolo_t[i:]
    if bolo_t[-1] > t1:
        i = np.argmin(np.fabs(bolo_t - t1))
        bolo_t = bolo_t[:i+1]
    bolo = []
    for t in bolo_t:
        i0 = np.argmin(np.fabs(kb5_t - t))
        i1 = np.argmin(np.fabs(kb5_t - (t + 2.*dt)))
        mean = np.mean(kb5[i0:i1+1], axis=0)
        bolo.append(mean)
        print('%10d %10.4f %10d %10d %10d %10.4f %10.4f' % (len(bolo), t, i0, i1, i1-i0+1, kb5_t[i0], kb5_t[i1]))
    bolo = np.array(bolo)
    print(pulse, 'bolo:', bolo.shape, bolo.dtype)
    print(pulse, 'bolo_t:', bolo_t.shape, bolo_t.dtype)
    return bolo, bolo_t
