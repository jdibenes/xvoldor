
import numpy as np
import struct

def flow_to_flo(flow_uv, filename):
    with open(filename, 'wb') as f:
        f.write('PIEH'.encode('ascii'))
        f.write(struct.pack('<II', flow_uv.shape[1], flow_uv.shape[0]))
        f.write(flow_uv.astype(np.float32).tobytes())
