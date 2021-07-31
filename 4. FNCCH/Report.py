'''
Report outputs from FNCCH reconstruction
@ Frankie Yeung (2021 May)

* calculations are reported in two files
* this is meant to be run AFTER calculations finish
> link_1.txt reports uni-directional links with format:
    i j tau* Cij(tau*)
    where i < j and Cij(tau*) has mean subtracted (normalized)
    if tau* > 0, we infer that a link points from i -> j
> link_2.txt reports bi-directional links with format:
    i j tau*(>0) Cij(tau*) tau*(<0) Cij(tau*)
    where i < j and Cij(tau*) has mean subtracted (normalized)
'''
import numpy as np
w = 200
out = 'out-w=%d/' % w
f1 = open('link_1.txt','w')
f2 = open('link_2.txt','w')
t = np.load(out+'t_phyfilter.npy')
c = np.load(out+'C_phyfilter.npy')
i = np.load(out+'idx_phyfilter.npy')
n = len(t)
j = 0
while j < n:
    a = min(i[0,j],i[1,j])
    b = max(i[0,j],i[1,j])
    if j+1<n and i[0,j]==i[1,j+1] and i[1,j]==i[0,j+1]:
        f2.write('%d %d %d %.10f %d %.10f\n' % (a,b,t[j]-w//2,c[j],t[j+1]-w//2,c[j+1]))
        j += 2
    else:
        f1.write('%d %d %d %.10f\n' % (a,b,t[j]-w//2,c[j]))
        j += 1
f1.close()
f2.close()
