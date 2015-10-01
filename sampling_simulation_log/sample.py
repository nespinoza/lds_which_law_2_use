import sys
import numpy as np
import matplotlib.pyplot as plt

mu = np.array([1e-70,1-1e-70])#np.append(np.arange(step,1,step),1.)

def check_kipping(A,B):
        I = 1. - A*(1.-mu)-B*mu*(1.-np.log(mu))
        dI = A+B*np.log(mu)
        if len(np.where(I<=0)[0])>0:
                return False
        else:
                if len(np.where(dI<=0)[0])>0:
                        return False
                else:
                        return True

def check_espinoza(l1,l2):
        I = 1. - l1*(1.-mu)-l2*mu*np.log(mu)
        dI = l1-l2*(1.+np.log(mu))
        if len(np.where(I<=0)[0])>0: 
                return False
        else:
                if len(np.where(dI<=0)[0])>0:
                        return False
                else:
                        return True

N = 10000
lims = 0.1
c1,c2 = np.random.uniform(0-lims,1+lims,N),np.random.uniform(0-lims,lims+1,N)

final_c1 = []
final_c2 = []
for i in range(N):
        if check_espinoza(c1[i],c2[i]):
                final_c1.append(c1[i])
                final_c2.append(c2[i])

import matplotlib.pyplot as plt
plt.xlabel('$l_1$')
plt.ylabel('$l_2$')
plt.style.use('ggplot')
plt.plot([-lims,1+lims],[-lims,1+lims],'-',label='$l_1>l_2$')
#plt.plot([-lims,lims],[-lims-2,lims-2],'-',label='$l_1-2<l_2$')
plt.plot([1,1],[-lims,lims+1],'-',label='$l_1<1$')
plt.plot([-lims,lims+1],[0,0],'-',label='$0<l_2$')
plt.plot([-lims,lims+1],[0,0],'-',label='$l_2>1$')
plt.plot(c1,c2,'r.',alpha=0.01)
plt.plot(np.array(final_c1),np.array(final_c2),'b.')
plt.legend()
plt.show()


        
