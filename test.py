import numpy as np
from Lineiterator import createLineIterator
y=[1,2]
hi=281
wi=500
iterlist=createLineIterator(np.array([0, round(hi * 0.80)]),np.array([wi, round(hi * 0.80)]))
#iterlist= [(x,hi-y) for (x,y) in iterlist]
totalUp=10
totalDown=15
status=200
info = [
        ("Exit", totalUp),
        ("Enter", totalDown),
        ("Status", status),
        ]
print(info[1][1])

print(y[-1])