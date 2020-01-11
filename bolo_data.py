
import sys
import h5py
import jetdata
import numpy as np

# ------------------------------------------------------------------------------

if len(sys.argv) < 2:
    print('Usage: %s pulse pulse ... (for individual pulses)' % sys.argv[0])
    print('Usage: %s pulse-pulse ... (for a range of pulses)' % sys.argv[0])
    exit()

pulses = []
for arg in sys.argv[1:]:
    if '-' in arg:
        parts = sorted(arg.split('-'))
        pulse0 = int(parts[0])
        pulse1 = int(parts[-1])
        for pulse in range(pulse0, pulse1+1):
            pulses.append(pulse)
    else:
        pulses.append(int(arg))

pulses = sorted(pulses)

# ------------------------------------------------------------------------------

fname = 'bolo_data.h5'
print('Writing:', fname)
f = h5py.File(fname, 'w')

for pulse in pulses:
    
    bolo, bolo_t = jetdata.get_bolo(pulse)
    if len(bolo) == 0:
        continue
    print('%-10s %-10s %-20s %-10s' % (pulse, 'bolo', bolo.shape, bolo.dtype))
    print('%-10s %-10s %-20s %-10s' % (pulse, 'bolo_t', bolo_t.shape, bolo_t.dtype))    

    g = f.create_group(str(pulse))
    g.create_dataset('bolo', data=bolo)
    g.create_dataset('bolo_t', data=bolo_t)
    print('-'*50)

f.close()
