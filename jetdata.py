
import ppf
import numpy as np

ppf.ppfssr([0,1,2,3,4])

def get_bolo(pulse):
    ppf.ppfgo(pulse)
    ppf.ppfuid('jetppf', 'r')
    ihdata, iwdata, kb5h, x, kb5h_t, ier = ppf.ppfget(pulse, 'bolo', 'kb5h', reshape=1)
    ihdata, iwdata, kb5v, x, kb5v_t, ier = ppf.ppfget(pulse, 'bolo', 'kb5v', reshape=1)
    bolo = np.hstack((kb5h, kb5v))
    assert np.all(kb5h_t == kb5v_t)
    bolo_t = kb5h_t
    return bolo, bolo_t

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
