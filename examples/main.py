from dmp import DynamicMovementPrimative as DMP
import numpy as np

mydmp = DMP(2,2,2,2,2)

#print(mydmp.distributions(mydmp.phase(np.linspace(1,2,50))))

data = DMP.load_demo("Demos/demo1.txt")

# test the following function
print(DMP.parse_demo(data))

q, t = DMP.parse_demo(data)

l = len(q);

t = DMP.normalize_vector(t)

def prototype(fileName):
    q, t = DMP.parseDemo(DMP.loadDemo(fileName))
    l = q.shape
    t = DMP.normalizeVector(t)
    
    
