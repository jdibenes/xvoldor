
import numpy as np
import struct

def flow_to_flo(flow_uv, filename):
    with open(filename, 'wb') as f:
        f.write('PIEH'.encode('ascii'))
        f.write(struct.pack('<II', flow_uv.shape[1], flow_uv.shape[0]))
        f.write(flow_uv.astype(np.float32).tobytes())

def flo_to_flow(filename):
    with open(filename, 'rb') as f:
        f.read(4)
        width, height = struct.unpack('<II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.float32).reshape((height, width, 2))
    
