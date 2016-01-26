# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from pyx import text as pyx_text
from pyx import trafo
def text_pyx(g, x_coord, y_coord, text_input, text_size = -2, color = None, rotation = 0.):
    """
    Function that draws text in a given plot
    INPUTS:
        g       (Object) A graph-type object to which you want to add the text
        x_coord     (Double) x-coordinate (in plot units) at which you want to place
                the text
        y_coord     (Double) y-coordinate (in plot units) at which you want to place
                the text
        text_input  (String) Text that you want to add.
        text_size   (int, optional) Text size of the text added to the plot. Default is -2.
        color       (instance) Color instance that defines the color that you want the 
                text to have. Default is black.
    """

    # First define the text attributes:
    textattrs = [pyx_text.size(text_size),pyx_text.halign.center, pyx_text.vshift.middlezero, trafo.rotate(rotation)]

    # Now convert plot positions to pyx's:
    x0,y0 = g.pos(x_coord, y_coord)

    # If no color is given, draw black text. If color is given, draw text with the input color:
    if color is None:
        g.text(x0,y0,text_input,textattrs)
    else:
        # First, check which was the input color palette:
        color_dict = color.color
        if len(color_dict.keys()) == 4:
            color_string = str(color_dict['c'])+','+str(color_dict['m'])+','+str(color_dict['y'])+','+str(color_dict['k'])
            color_palette = 'cmyk'
        else:
                        color_string = str(color_dict['r'])+','+str(color_dict['g'])+','+str(color_dict['b'])
                        color_palette = 'rgb'
        # Now draw the text:
        g.text(x0, y0, r"\textcolor["+color_palette+"]{"+color_string+"}{"+text_input+"}",textattrs)

from pyx import *
results = 'results/sic/'
ld_laws = ['linear','quadratic','squareroot','logarithmic','three-param']
names = ['Linear law', 'Quadratic law', 'Square-root law', 'Logarithmic law','Three-parameter law']
rotation = [0.0,11.,3.0,9.,35]
delta_y = [3e-7,0.001e-9,1.5e-8,0.12e-8,-6.25e-9]
colors = [color.cmyk.Grey,color.cmyk.Black,color.cmyk.Orange,color.cmyk.CornflowerBlue,color.cmyk.OliveGreen]
N = [100,500,1000]
precisions = np.array([10.,20.,30.,40.,50.,60.,70.,80.,90.,100.,200.,300.,\
                       400.,500.,600.,700.,800.,900.,1000.,2000.,3000.])

data = {}
for ld_law in ld_laws:
    data[ld_law] = {}
    for j in range(len(N)):
        data[ld_law][str(N[j])] = {}
        biases_p = np.arange(len(precisions)).astype('float64')
        precisions_p = np.arange(len(precisions)).astype('float64')
        for i in range(len(precisions)):
            bias_p,precision_p,bias_a,precision_a,bias_i,precision_i = np.loadtxt(results+'/N_'+str(N[j])+'_precision_'+\
                                                                   str(precisions[i])+'/results_'+ld_law+'.dat')
            biases_p[i] = bias_p
            precisions_p[i] = precision_p
        data[ld_law][str(N[j])]['precision_p'] = np.copy(precisions_p)
        data[ld_law][str(N[j])]['bias_p'] = np.copy(biases_p)
        
#    plt.plot(precisions,np.abs(biases_p)/precisions_p,'.-',label=ld_law)
#    plt.legend()
#plt.plot(precisions,np.ones(len(precisions)),'--')
#plt.xlabel('Lightcurve noise level (ppm)')
#plt.ylabel(r'$\Delta p/\sigma_p$ (Bias/Precision)')
#plt.yscale('log')
#plt.xscale('log')
#plt.show()

# Generate the plot:
# Define plot options:
unit.set(xscale = 0.8)
text.set(mode="latex")
text.preamble(r"\usepackage{color}")
text.preamble(r"\usepackage{wasysym}")
text.preamble(r"\usepackage{mathabx}")
legend_text_size = -1
legend_pos = 'tl'
min_y= 1e-9#5e-3#0.8
max_y= 2e-5#50.#4.3
min_x = 9.#0.7
max_x = 3500.#32.
#xaxis = ''
#yaxis = ''
xaxis = r'Lightcurve noise level (ppm)'
yaxis = r'MSE = Bias$^2$ + Variance'
outname = 'bias_variance_tradeoff_mse'

# Plot
c = canvas.canvas()
g = c.insert(graph.graphxy(height=10,width=17,
        key=graph.key.key(pos=legend_pos,textattrs=[text.size(legend_text_size)]),\
        x = graph.axis.logarithmic(min=min_x,max=max_x,title = xaxis,
                texter=graph.axis.texter.decimal()),\
        y = graph.axis.logarithmic(min=min_y,max=max_y,title = yaxis)))

#g.plot(graph.data.values(x=[1.,1e4],y=[1.,1.], title = None),\
#                             styles = [graph.style.line([color.cmyk.Grey,\
#                                       style.linestyle.dashed,\
#                                       style.linewidth.thin])])

# Plot fake lines to put labels:
g.plot(graph.data.values(x=[5000,10000],y=[1,1], title = '100 in-transit points'),\
                styles = [graph.style.line([color.cmyk.Black,\
                style.linestyle.solid,\
                style.linewidth.THick])])

g.plot(graph.data.values(x=[5000,10000],y=[1,1], title = '1000 in-transit points'),\
                styles = [graph.style.line([color.cmyk.Black,\
                style.linestyle.dashed,\
                style.linewidth.thick])])

for i in range(len(ld_laws)):
    ld_law = ld_laws[i]
    g.plot(graph.data.values(x=precisions,y=data[ld_law]['100']['bias_p']**2+data[ld_law]['100']['precision_p']**2, title = None),\
                             styles = [graph.style.line([colors[i],\
                                       style.linestyle.solid,\
                                       style.linewidth.THick])])

    print 'ld_law     : ',ld_law
    print 'lc precs   : ',precisions
    print 'biases     : ',data[ld_law]['100']['bias_p']
    print 'precisions : ',data[ld_law]['100']['precision_p']

    text_pyx(g, np.sqrt(precisions[0]*precisions[1]), data[ld_law]['100']['bias_p'][1]**2+data[ld_law]['100']['precision_p'][1]**2+delta_y[i], names[i], \
             text_size = -1, color = colors[i], rotation = rotation[i])

    g.plot(graph.data.values(x=precisions,y=data[ld_law]['1000']['bias_p']**2+data[ld_law]['1000']['precision_p']**2, title = None),\
                             styles = [graph.style.line([colors[i],\
                                       style.linestyle.dashed,\
                                       style.linewidth.thick])])

#text_pyx(g, , 2000., names[i])
c.writeEPSfile(outname,write_mesh_as_bitmap = True,write_mesh_as_bitmap_resolution=5)
c.writePDFfile(outname,write_mesh_as_bitmap = True,write_mesh_as_bitmap_resolution=5)
