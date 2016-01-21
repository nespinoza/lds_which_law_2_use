import numpy as np

data = np.genfromtxt('data.tab',usecols=(4,22,23,24),\
                dtype=None,delimiter='\t')

error_treshold = 0.01 # percent
all_candidates = 0
all_confirmed = 0
candidate_count = 0
confirmed_count = 0
for i in range(len(data)):
    status,depth,depth_err_up,depth_err_down = data[i]
    if not np.isnan(depth) and not np.isnan(depth_err_up) and not np.isnan(depth_err_down):
        # Get planet-to-star radius ratio as sqrt(depth):
        p = np.sqrt(depth*1e-6)
        # Estimate error on this via delta method:
        perr = np.max([depth_err_up*1e-6,np.abs(depth_err_down*1e-6)])/(2.*np.sqrt(np.sqrt(depth*1e-6)))
        # Get quoted precision:
        precision = (perr/p)*100
        if precision < error_treshold:
                if status == 'CANDIDATE':
                   candidate_count = candidate_count + 1
                if status == 'CONFIRMED':
                   confirmed_count = confirmed_count + 1
        if status == 'CANDIDATE':
                all_candidates = all_candidates + 1
        if status == 'CONFIRMED':
                all_confirmed = all_confirmed + 1
print '\t Out of '+str(all_candidates)+' planet candidates:'
print '\t > A total of '+str(candidate_count)+' have precision better than '+str(error_treshold)+'%.'
print '\t Out of '+str(all_confirmed)+' confirmed exoplanets:'
print '\t > A total of '+str(confirmed_count)+' have precision better than '+str(error_treshold)+'%.'

#print depth
#print depth_err_up
#print len(depth)
