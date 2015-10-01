from pyx import *
import numpy as np
import matplotlib.pyplot as plt

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

def draw_text(g, x_coord, y_coord, text_input, text_size = -2):
    textattrs = [text.size(text_size),text.halign.center, text.vshift.middlezero]
    x0,y0 = g.pos(x_coord, y_coord)
    g.text(x0,y0,text_input,textattrs)

def draw_rectangular_band(g, x_coord_init, y_coord_init, w, h, band_color):

    x0,y0 = g.pos(x_coord_init , y_coord_init)
    xf,yf = g.pos(x_coord_init+w, y_coord_init+h)

    x = x0
    y = y0
    w = np.abs(x0-xf)
    h = np.abs(y0-yf)

    g.fill(path.rect(x,y,w,h), [band_color])

def get_color(teff,i,half_val,m,n):
    if teff[i] < half_val:
        c_color = 2.0*(m*(teff[i])+n)
        m_color = 1.-2.0*(m*(teff[i])+n)
        y_color = 1.0
        k_color = 0.
    else:
        c_color = 1.0
        m_color = (m*(teff[i])+n) - 0.5
        y_color = 2.0-2.0*(m*(teff[i])+n)
        k_color = 0.

    return color.cmyk(c_color,m_color,y_color,k_color)        

teff,logg,mh,vturb,c1,c2 = np.loadtxt('all_atlas_lds_kepler.dat',unpack=True,usecols = (3,4,5,6,19,20))

idx = np.where((teff<9000.))[0]
mh = mh[idx]
vturb = vturb[idx]
logg = logg[idx]
teff = teff[idx]
c1 = c1[idx]
c2 = c2[idx]

######## PLOTTING ###########
# Now plot. First, some options:
unit.set(xscale = 0.8)
text.set(mode="latex")
text.preamble(r"\usepackage{color}")
text.preamble(r"\usepackage{wasysym}")
legend_text_size = -2
min_x = 0.2#np.min(c1)
max_x = 0.9#np.max(c1)
min_y= -0.005#np.min(c2)
max_y= 0 #np.max(c2)
print 'min/max e1:',min_x,max_x
print 'min/max e2:',min_y,max_y
# More options on the legend:
legend_pos = 'br'
xaxis = r'$e_1$'
yaxis = r'$e_2$'
outname = 'exponential_ldcs'

c = canvas.canvas()
g = c.insert(graph.graphxy(height=7,width=7, 
        key=graph.key.key(pos=legend_pos,textattrs=[text.size(legend_text_size)]),\
        x = graph.axis.linear(min=min_x,max=max_x,title = xaxis, 
                texter=graph.axis.texter.decimal()),\
        y = graph.axis.linear(min=min_y,max=max_y,title = yaxis)))

# Get minimum and maximum teff:
min_val = np.min(teff)
max_val = np.max(teff)
half_val = min_val + ((max_val - min_val)/2.)
m = 1./(max_val-min_val)
n = -min_val*m
# Plot LDCs
loggs = []
mhs = []
for i in range(len(c1)):
    if logg[i] not in loggs:
            loggs.append(logg[i])
    if mh[i] not in mhs:
            mhs.append(mh[i])
    #c_color = 1.-(m*(teff[i])+n)  # This is 1 with smaller Teff
    #m_color = m*(teff[i])+n       # This is 0 with smaller Teff
    #y_color = m*(teff[i])+n
    #k_color = 0.0
    # First, go from red (0,1,1,0) to green (1,0,1,0), then go from green to blue (1,0.5,0,0)
    #if teff[i] < half_val: 
    #    c_color = 2.0*(m*(teff[i])+n)
    #    m_color = 1.-2.0*(m*(teff[i])+n)
    #    y_color = 1.0
    #    k_color = 0.
    #else:
    #    c_color = 1.0
    #    m_color = (m*(teff[i])+n) - 0.5
    #    y_color = 2.0-2.0*(m*(teff[i])+n)
    #    k_color = 0.
    the_color = get_color(teff,i,half_val,m,n)
    g.plot(graph.data.values(x=[c1[i]], y=[c2[i]], title = None),\
            styles = [graph.style.symbol(graph.style.symbol.circle,\
            symbolattrs = [deco.filled([the_color]),  
                        deco.stroked([the_color])],\
                        size = 0.03)])

###################### COLORBAR EMULATION ##################################################
# Define plot that will emulate the colorbar. First, define a fake y-axis:
mymanualticks = [graph.axis.tick.tick(-2,label=""),graph.axis.tick.tick(-1,label="")]

# Now a true y-axis with values I want to show:
value_range = [0,1,2,3]
tick_range = [4000,5000,6000,7000,8000]
paint_tick_range = np.linspace(np.min(teff),np.max(teff),100)
paint_value_range = np.double(np.arange(len(paint_tick_range)))
paint_value_range = (paint_value_range/paint_value_range[-1])*3.0
mymanualticks_x = []
mymanualticks_x2 = []
for i in range(len(value_range)):
        mymanualticks_x.append(graph.axis.tick.tick(value_range[i],label=str(tick_range[i])))
        mymanualticks_x2.append(graph.axis.tick.tick(value_range[i],label=""))

cbar = c.insert(graph.graphxy(height=0.5,width=7, ypos = g.ypos + 7.5, \
                                                  xpos = g.xpos,\
       key=graph.key.key(pos=legend_pos,textattrs=[text.size(-3)]),\
       y = graph.axis.linear(min=-2,max=-1,title = None,manualticks=mymanualticks),\
       x2 = graph.axis.linear(min=np.min(value_range)-0.5,max=np.max(value_range)+0.5,\
                              manualticks = mymanualticks_x,title = r'$T_\textnormal{eff}$',parter=None)))

# "Paint" the plot:

for i in range(len(paint_tick_range)):
   the_color = get_color(paint_tick_range,i,half_val,m,n)
   draw_rectangular_band(cbar, paint_value_range[i] - 0.5, -2, 1., 1.0, the_color)

# Plot borders of cb
cbar = c.insert(graph.graphxy(height=0.5,width=7, ypos = g.ypos + 7.5, \
                                                  xpos = g.xpos,\
       key=graph.key.key(pos=legend_pos,textattrs=[text.size(-3)]),\
       y = graph.axis.linear(min=-2,max=-1,title = None,manualticks=mymanualticks),\
       x2 = graph.axis.linear(min=np.min(value_range)-0.5,max=np.max(value_range)+0.5,\
                              manualticks = mymanualticks_x2,title = None,parter=None)))
###########################################################################################



# Plot e2 < 0 constrain (everywhere intensity profile):
#for grav in loggs:
#    idx_g = np.where(logg==grav)[0]
#    g.plot(graph.data.values(x=c1[idx_g], y=c2[idx_g], title = None),\
#                       styles = [graph.style.line([color.cmyk.Grey,\
#                                 style.linestyle.solid,\
#                                 style.linewidth.thin])])
#    draw_text(g, np.min(c1[idx_g]), np.min(c2[idx_g]), '$log(g) ='+str(grav)+'$', text_size = -2)

#for met in mhs:
#    idx_g = np.where(mh==met)[0]
#    g.plot(graph.data.values(x=c1[idx_g], y=c2[idx_g], title = None),\
#                       styles = [graph.style.line([color.cmyk.Grey,\
#                                 style.linestyle.solid,\
#                                 style.linewidth.thin])])
#    draw_text(g, np.min(c1[idx_g]), np.min(c2[idx_g]), '$[M/H] ='+str(met)+'$', text_size = -2)

# Plot arrows indicating the constrains:
#draw_arrow(g, min_x, 0, min_x, (max_y/2.), 0.05, color.cmyk.Red)
#draw_text(g, min_x + 0.1, 0, r'$e_2>0$ ($I(\mu)>0$)')

c.writeEPSfile(outname,write_mesh_as_bitmap = True,write_mesh_as_bitmap_resolution=5)
c.writePDFfile(outname,write_mesh_as_bitmap = True,write_mesh_as_bitmap_resolution=5)
