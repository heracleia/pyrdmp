from DMP_class import DynamicMovementPrimative as DMP
import numpy as np

mydmp = DMP(2,2,2,2,2)

#print(mydmp.distributions(mydmp.phase(np.linspace(1,2,50))))

data = DMP.loadDemo("Demos/demo1.txt")

# test the following function
print(DMP.parseDemo(data))

q, t = DMP.parseDemo(data)

l = len(q);

t = DMP.normalizeVector(t)

def prototype(fileName):
    q, t = DMP.parseDemo(DMP.loadDemo(fileName))
    l = q.shape
    t = DMP.normalizeVector(t)
    
    dq = np.zeros(
