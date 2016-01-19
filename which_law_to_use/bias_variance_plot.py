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

def draw_arrow(g, x_coord_init, y_coord_init, x_coord_final, y_coord_final, size, line_color, stroke_color = None, fill_color = None):

    if stroke_color is None:
       stroke_color = line_color
    if fill_color is None:
       fill_color = line_color

    x0,y0 = g.pos(x_coord_init , y_coord_init)
    xf,yf = g.pos(x_coord_final, y_coord_final)
    g.stroke(path.line(x0,y0,xf,yf),\
         [style.linewidth.normal, style.linestyle.solid, line_color,
          deco.earrow([deco.stroked([stroke_color]),
                       deco.filled([fill_color])], size=0.1)])

from pyx import *
x = np.linspace(-10,10,1000)
mu = 4.0
sigma = 4.0
y = (1./np.sqrt(2.*np.pi*sigma))*np.exp(-(x-mu)**2/(2.*sigma**2))


# Generate the plot:
# Define plot options:
unit.set(xscale = 0.9)
text.set(mode="latex")
text.preamble(r"\usepackage{color}")
text.preamble(r"\usepackage{wasysym}")
text.preamble(r"\usepackage{mathabx}")
text.preamble(r"\usepackage{amsfonts}")
legend_text_size = -1
legend_pos = 'tr'
min_y= 0.0#0.8
max_y= np.max(y)+0.1#4.3
min_x = -10#0.7
max_x = 10#32.
#xaxis = ''
#yaxis = ''
xaxis = r'Parameter value'
yaxis = r'Probability density'
outname = 'bias_variance_plot'

# Plot
c = canvas.canvas()
ticks = [graph.axis.tick.tick(0, label=r"$\theta$"),
         graph.axis.tick.tick(mu, label=r"\color{red}{$\mathbb{E}[\hat{\theta}]$}")]

g = c.insert(graph.graphxy(height=7,width=10,
        key=graph.key.key(pos=legend_pos,textattrs=[text.size(legend_text_size)]),\
        x = graph.axis.linear(min=min_x,max=max_x,title = xaxis,manualticks=ticks,parter=None),\
        y = graph.axis.linear(min=min_y,max=max_y,title = yaxis)))

g.plot(graph.data.values(x=x,y=y, title = None),\
                styles = [graph.style.line([color.cmyk.Red,\
                style.linestyle.solid,\
                style.linewidth.thick])])

g.plot(graph.data.values(x=[mu,mu],y=[0.,1e3], title = None),\
                styles = [graph.style.line([color.cmyk.Red,\
                style.linestyle.dashed,\
                style.linewidth.thick])])

g.plot(graph.data.values(x=[0,0],y=[0.,1e3], title = None),\
                styles = [graph.style.line([color.cmyk.Black,\
                style.linestyle.solid,\
                style.linewidth.thick])])

draw_arrow(g,0.+0.15,0.25,mu-0.15,0.25,1.0,color.cmyk.Black)
draw_arrow(g,mu+0.15,(1./np.sqrt(2.*np.pi*sigma))*np.exp(-(sigma)**2/(2.*sigma**2)),mu+sigma-0.15,\
                (1./np.sqrt(2.*np.pi*sigma))*np.exp(-(sigma)**2/(2.*sigma**2)),1.0,color.cmyk.Red)

text_pyx(g,mu+(sigma/2.),(1./np.sqrt(2.*np.pi*sigma))*np.exp(-(sigma)**2/(2.*sigma**2))+0.01,r'$\sqrt{\textnormal{Var}[\hat{\theta}]}$',color=color.cmyk.Red)
text_pyx(g,mu+(sigma/2.),(1./np.sqrt(2.*np.pi*sigma))*np.exp(-(sigma)**2/(2.*sigma**2))-0.01,r'(1/Precision)',color=color.cmyk.Red)
text_pyx(g,mu/2.,0.25 + 0.01,r'$\mathbb{E}[\hat{\theta}]-\theta$')
text_pyx(g,mu/2.,0.25 - 0.01,r'(Bias)')

c.writeEPSfile(outname,write_mesh_as_bitmap = True,write_mesh_as_bitmap_resolution=5)
c.writePDFfile(outname,write_mesh_as_bitmap = True,write_mesh_as_bitmap_resolution=5)
