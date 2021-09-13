import numpy as np
from Lineiterator import createLineIterator
y=[1,2,3,4,5,5,7,8,9,10]
hi=281
wi=500
iterlist=createLineIterator(np.array([0, round(hi * 0.80)]),np.array([wi, round(hi * 0.80)]))
#iterlist= [(x,hi-y) for (x,y) in iterlist]
print(iterlist)