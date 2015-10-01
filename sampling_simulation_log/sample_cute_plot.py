from pyx import *
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
        if l1<1 and l2>0 and l1>l2:
                return True
        return False

def check_espinoza2(l1,l2):
        I = 1. - l1*(1.-mu)-l2*mu*np.log(mu)
        dI = l1-l2*(1.+np.log(mu))
        if len(np.where(I<=0)[0])>0: 
                return False
        else:
                if len(np.where(dI<=0)[0])>0:
                        return False
                else:
                        return True

N = 1000000
lims = 1
c1,c2 = np.random.uniform(0-lims,1+lims,N),np.random.uniform(0-lims,lims+1,N)

final_c1 = []
final_c2 = []
for i in range(N):
        if check_espinoza(c1[i],c2[i]):
                final_c1.append(c1[i])
                final_c2.append(c2[i])

print len(final_c1)
print len(c1)
print 100*(np.double(len(final_c1))/np.double(len(c1)))
######## PLOTTING ###########
# Now plot. First, some options:
unit.set(xscale = 0.8)
text.set(mode="latex")
text.preamble(r"\usepackage{color}")
text.preamble(r"\usepackage{wasysym}")
legend_text_size = -2
min_x = 0-lims
max_x = 1+lims#0.55
min_y= 0-lims
max_y= 1+lims#0.75#1.05
# More options on the legend:
legend_pos = 'br'
xaxis = r'$l_1$'
yaxis = r'$l_2$'
outname = 'triangular_sampling'

c = canvas.canvas()
g = c.insert(graph.graphxy(height=7,width=7, 
        key=graph.key.key(pos=legend_pos,textattrs=[text.size(legend_text_size)]),\
        x = graph.axis.linear(min=min_x,max=max_x,title = xaxis, 
                texter=graph.axis.texter.decimal()),\
        y = graph.axis.linear(min=min_y,max=max_y,title = yaxis)))

# Plot constrains:
g.plot(graph.data.values(x=[-lims,1+lims], y=[-lims,1+lims], title = None),\
                       styles = [graph.style.line([color.cmyk.Red,\
                                 style.linestyle.dashed,\
                                 style.linewidth.thin])])

g.plot(graph.data.values(x=[1,1], y=[-lims,lims+1], title = None),\
                       styles = [graph.style.line([color.cmyk.Red,\
                                 style.linestyle.dashed,\
                                 style.linewidth.thin])])

g.plot(graph.data.values(x=[-lims,lims+1],y=[0,0], title = None),\
                       styles = [graph.style.line([color.cmyk.Red,\
                                 style.linestyle.dashed,\
                                 style.linewidth.thin])])

# Plot sampled points:
#g.plot(graph.data.values(x=c1, y=c2, title = None),\
#        styles = [graph.style.symbol(graph.style.symbol.circle,\
#        symbolattrs = [deco.filled([color.cmyk.Gray]), 
#                       deco.stroked([color.cmyk.Gray])],\
#                       size = 0.01)])

g.plot(graph.data.values(x=final_c1[:2000], y=final_c2[:2000], title = None),\
        styles = [graph.style.symbol(graph.style.symbol.circle,\
        symbolattrs = [deco.filled([color.cmyk.Gray]),  
                       deco.stroked([color.cmyk.Gray])],\
                       size = 0.01)])


c.writeEPSfile(outname)
c.writePDFfile(outname)
