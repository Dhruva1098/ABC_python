import numpy as np
import numpy.matlib
import random


def sphere(x):
    y = np.empty((10,1))
    y[:] = np.NaN
    
    
    for i in range(len(x)):
        z = 0
        t = 0
        row = x[i, :]
        t = np.square(row)
        z = np.sum(t)
        y[i]=z
    return(y)  
        
def CalFit(f):
    if f >= 0:
        fit = 1/(1+f)
    else:
        fit = 1 + abs(f)
    return fit

lb = np.array([0,0])
ub = np.array([500,500])
Np = 10;
T = 50;
limit = 5;
t=0
f = np.empty((Np,1))
f[:] = np.NaN
fit = np.empty((Np,1))
fit[:] = np.NaN
trial = np.empty((Np,1))

D = len(lb)


rand = np.random.rand(Np,D)

arr1 = np.matlib.repmat(lb,Np,1)
arr2 = np.matlib.repmat(ub-lb,Np,1)
P = arr1 +arr2*rand

f= sphere(P)
for p in range(Np):
     fit[p]=CalFit(f[p])

bestobj = min(f)
ind = f.argmin()
bestsol = P[ind]

def GenNewSol(lb,ub,Np,n,P,fit,trial,f,D):
    j = random.randint(0,D-1)
    p = random.randint(0,Np-1)
    
    while(p==n):
        p = random.randint(0,Np-1)
       
    Xnew = P[n]
    
    phi = -1 + (1-(-1))*random.uniform(-1,1)
    
    Xnew[j] = P[n,j] + phi*(P[n,j] - P[p,j])
    
    Xnew[j] = min(Xnew[j],ub[j])
    
    Xnew[j] = max(Xnew[j],lb[j])
    
    

        
    z = np.square(Xnew)
    ObjNewBol = np.sum(z)
    FitnessNewSol = CalFit(ObjNewBol)
    
    
    if(FitnessNewSol>fit[n]):
        P[n] = Xnew
        fit[n] = FitnessNewSol
        f[n]=ObjNewBol
        trial[n]=0
        
    else:
        trial[n] = trial[n] + 1
        
        
    return trial, P, fit, f

while(t<T):
    #employed
    for i in range(Np):
        trial,P,fit,f = GenNewSol(lb,ub,Np,i,P,fit,trial,f,D)
    
    
    
    #onlooker bee
    probability = 0.9*(fit/max(fit)) + 0.1
    m = 0
    n = 1
    while(m<Np):
        i = 0
        if (random.random() < probability[i]):
            trial,P,fit,f = GenNewSol(lb,ub,Np,i,P,fit,trial,f,D)
            m = m+1
        else:
            n = n%Np + 1
        i += 1
    stack = np.vstack([f,bestobj])
    bestobj = min(stack)
    ind = stack.argmin()
    CombinedSol = np.vstack([P,bestsol])
    bestsol = CombinedSol[ind]
    
    
    #scout bee
    val = max(trial)
    ind = trial.argmax()

    if(val>limit):
        trial[ind]=0;
        P[ind] = lb + (ub - lb)*np.random.rand(1,D)
        z = np.square(P[ind])
        F = np.sum(z)
        f[ind]=F
        fit[ind]=CalFit(f[ind])
        
    stack = np.vstack([f,bestobj])
    bestobj = min(stack)
    ind = stack.argmin()
    CombinedSol = np.vstack([P,bestsol])
    bestsol = CombinedSol[ind]
    
    
    
    t+=1


    
