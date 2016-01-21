from pyx import *
import numpy as np

import numpy as np

def draw_rectangular_band(g, x_coord_init, y_coord_init, w, h, band_color):

    x0,y0 = g.pos(x_coord_init , y_coord_init)
    xf,yf = g.pos(x_coord_init+w, y_coord_init+h)

    x = x0
    y = y0
    w = np.abs(x0-xf)
    h = np.abs(y0-yf)

    g.fill(path.rect(x,y,w,h), [band_color])

def draw_text(g, x_coord, y_coord, text_input, text_size = 1):
    textattrs = [text.size(text_size),text.halign.center, text.vshift.middlezero]
    x0,y0 = g.pos(x_coord, y_coord)
    g.text(x0,y0,text_input,textattrs)

b = 0.0
ytext = 0.35
p_scale = 0.1
p_scale2 = 2.5
p_show = [0.01,0.07,0.13]
fit_option = 'float'

# Get the results:
grid_num_tpl, p_tpl, a_tpl, input_teff_tpl, p_fixed_tpl, sigma_p_fixed_tpl, p_float_tpl, sigma_p_float_tpl, \
                a_fixed_tpl, sigma_a_fixed_tpl, a_float_tpl, sigma_a_float_tpl,i_fixed_tpl, sigma_i_fixed_tpl, i_float_tpl, sigma_i_float_tpl \
                = np.loadtxt('threeparam_results/final_results.dat',unpack=True)

grid_num_log, p_log, a_log, input_teff_log, p_fixed_log, sigma_p_fixed_log, p_float_log, sigma_p_float_log, \
                a_fixed_log, sigma_a_fixed_log, a_float_log, sigma_a_float_log,i_fixed_log, sigma_i_fixed_log, i_float_log, sigma_i_float_log \
                = np.loadtxt('logarithmic_results/final_results.dat',unpack=True)

grid_num_sqrt, p_sqrt, a_sqrt, input_teff_sqrt, p_fixed_sqrt, sigma_p_fixed_sqrt, p_float_sqrt, sigma_p_float_sqrt, \
                a_fixed_sqrt, sigma_a_fixed_sqrt, a_float_sqrt, sigma_a_float_sqrt,i_fixed_sqrt, sigma_i_fixed_sqrt, i_float_sqrt, sigma_i_float_sqrt \
                = np.loadtxt('squareroot_results/final_results.dat',unpack=True)

grid_num_quad, p_quad, a_quad, input_teff_quad, p_fixed_quad, sigma_p_fixed_quad, p_float_quad, sigma_p_float_quad, \
                a_fixed_quad, sigma_a_fixed_quad, a_float_quad, sigma_a_float_quad,i_fixed_quad, sigma_i_fixed_quad, i_float_quad, sigma_i_float_quad \
                = np.loadtxt('quadratic_results/final_results.dat',unpack=True)

grid_num_lin, p_lin, a_lin, input_teff_lin, p_fixed_lin, sigma_p_fixed_lin, p_float_lin, sigma_p_float_lin, \
                a_fixed_lin, sigma_a_fixed_lin, a_float_lin, sigma_a_float_lin,i_fixed_lin, sigma_i_fixed_lin, i_float_lin, sigma_i_float_lin \
                = np.loadtxt('linear_results/final_results.dat',unpack=True)

# Order the results from largest to smallest p (in order to have the smallest points on top of larger ones):
variables = ['grid_num_', 'p_', 'a_', 'input_teff_', 'p_'+fit_option+'_', 'sigma_p_'+fit_option+'_', 
                                'a_'+fit_option+'_', 'sigma_a_'+fit_option+'_', 'i_'+fit_option+'_', 'sigma_i_'+fit_option+'_']

################################# PLOTTING ######################################################
# Now plot. First, some options:
pheight = 3#5
pwidth = 4#6.5
delta_x = 5.0/1.2#9
delta_y = 1.0
unit.set(xscale = 1.0)
text.set(mode="latex")
text.preamble(r"\usepackage{color}")
text.preamble(r"\usepackage{wasysym}")
legend_text_size = -3
# More options on the legend:
legend_pos = 'tl'
xaxis = r'Host star $T_{\rm{eff}}$ (K)'
min_x = 3200.0
max_x = 9000.0
min_y_p = 0.0#-10
max_y_p = 1.0#1
max_y_p_2 = 0.32
min_y_p_2 = -0.32
min_y_a = -0.5#-2
max_y_a = 2.0#15
min_y_a_2 = -0.32
max_y_a_2 = 0.32
#min_y_i = -7.5
#max_y_i = 1.0
#min_y_i_2 = -2.3
#max_y_i_2 = 1.3
yaxis_p = r'$(\hat{p} - p)/p\ (\%)$'
yaxis_a = r'$(\hat{a}_R - a_R)/a_R\ (\%)$'
#yaxis_i = r'$(\hat{i} - i)/i\ (\%)$'
outname = 'simulation_b_'+str(b).split('.')[0]+str(b).split('.')[1]+'_'+fit_option

# And now the real plotting. First, define the plots for the biases on each parameter:
c = canvas.canvas()

# Plot for bias on different parameters for the linear law:
g_a_lin = c.insert(graph.graphxy(height=pheight,width=pwidth, \
       key=graph.key.key(pos=legend_pos,textattrs=[text.size(legend_text_size)]),\
       x = graph.axis.linear(min=min_x,max=max_x,title = xaxis, texter=graph.axis.texter.decimal()),\
       y = graph.axis.linear(min=min_y_a,max=max_y_a,title = yaxis_a)))

g_p_lin = c.insert(graph.graphxy(height=pheight,width=pwidth, ypos = g_a_lin.ypos+g_a_lin.height+delta_y,\
       key=graph.key.key(pos=legend_pos,textattrs=[text.size(legend_text_size)]),\
       x = graph.axis.linkedaxis(g_a_lin.axes["x"]),\
       y = graph.axis.linear(min=min_y_p,max=max_y_p,title = yaxis_p)))

# Plot for bias on different parameters for the quadratic law:
g_a_quad = c.insert(graph.graphxy(height=pheight,width=pwidth, xpos = g_a_lin.xpos + 1.2*delta_x+0.5, \
       key=graph.key.key(pos=legend_pos,textattrs=[text.size(legend_text_size)]),\
       x = graph.axis.linear(min=min_x,max=max_x,title = xaxis, texter=graph.axis.texter.decimal()),\
       y = graph.axis.linear(min=min_y_a_2,max=max_y_a_2,title=None)))

g_p_quad = c.insert(graph.graphxy(height=pheight,width=pwidth, xpos = g_a_lin.xpos + 1.2*delta_x+0.5, \
                                                               ypos = g_a_quad.ypos+g_a_quad.height+delta_y,\
       key=graph.key.key(pos=legend_pos,textattrs=[text.size(legend_text_size)]),\
       x = graph.axis.linkedaxis(g_a_quad.axes["x"]),\
       y = graph.axis.linear(min=min_y_p_2,max=max_y_p_2,title = None)))

# Same for logarithmic law:
g_a_log = c.insert(graph.graphxy(height=pheight,width=pwidth, xpos = g_a_quad.xpos + delta_x, \
       key=graph.key.key(pos=legend_pos,textattrs=[text.size(legend_text_size)]),\
       x = graph.axis.linear(min=min_x,max=max_x,title = xaxis, texter=graph.axis.texter.decimal()),\
       y = graph.axis.linkedaxis(g_a_quad.axes["y"])))

g_p_log = c.insert(graph.graphxy(height=pheight,width=pwidth, xpos = g_a_quad.xpos + delta_x, \
                                                                           ypos = g_a_log.ypos+g_a_log.height+delta_y,\
       key=graph.key.key(pos=legend_pos,textattrs=[text.size(legend_text_size)]),\
       x = graph.axis.linkedaxis(g_a_log.axes["x"]),\
       y = graph.axis.linkedaxis(g_p_quad.axes["y"])))

# And for square-root law:
g_a_sqrt = c.insert(graph.graphxy(height=pheight,width=pwidth, xpos = g_a_log.xpos + delta_x, \
       key=graph.key.key(pos=legend_pos,textattrs=[text.size(legend_text_size)]),\
       x = graph.axis.linear(min=min_x,max=max_x,title = xaxis, texter=graph.axis.texter.decimal()),\
       y = graph.axis.linkedaxis(g_a_quad.axes["y"])))

g_p_sqrt = c.insert(graph.graphxy(height=pheight,width=pwidth, xpos = g_a_log.xpos + delta_x, \
                                                                            ypos = g_a_sqrt.ypos+g_a_sqrt.height+delta_y,\
       key=graph.key.key(pos=legend_pos,textattrs=[text.size(legend_text_size)]),\
       x = graph.axis.linkedaxis(g_a_sqrt.axes["x"]),\
       y = graph.axis.linkedaxis(g_p_quad.axes["y"])))

# And three-parameter law:
g_a_tpl = c.insert(graph.graphxy(height=pheight,width=pwidth, xpos = g_a_sqrt.xpos + delta_x, \
       key=graph.key.key(pos=legend_pos,textattrs=[text.size(legend_text_size)]),\
       x = graph.axis.linear(min=min_x,max=max_x,title = xaxis, texter=graph.axis.texter.decimal()),\
       y = graph.axis.linkedaxis(g_a_quad.axes["y"])))

g_p_tpl = c.insert(graph.graphxy(height=pheight,width=pwidth, xpos = g_a_sqrt.xpos + delta_x, \
                                                              ypos = g_a_tpl.ypos+g_a_tpl.height+delta_y,\
       key=graph.key.key(pos=legend_pos,textattrs=[text.size(legend_text_size)]),\
       x = graph.axis.linkedaxis(g_a_tpl.axes["x"]),\
       y = graph.axis.linkedaxis(g_p_quad.axes["y"])))

###################### COLORBAR EMULATION ##################################################
"""
# Define plot that will emulate the colorbar. First, define a fake x-axis:
mymanualticks = [graph.axis.tick.tick(-2,label=""),graph.axis.tick.tick(-1,label="")]

# Now a true y-axis with values I want to show:
value_range = [0,1,2,3,4,5,6]
tick_range = [3.27, 3.92, 4.87, 6.45, 9.52, 18.18,200]
mymanualticks_y = []
mymanualticks_y2 = []
for i in range(len(value_range)):
        mymanualticks_y.append(graph.axis.tick.tick(value_range[i],label=str(tick_range[i])))
        mymanualticks_y2.append(graph.axis.tick.tick(value_range[i],label=""))

cbar = c.insert(graph.graphxy(height=pheight*1.5,width=pwidth/4., xpos = g_i_tpl.xpos + 1.35*delta_x, \
                                                                  ypos = g_i_tpl.ypos+g_i_tpl.height+2.*delta_y,\
       key=graph.key.key(pos=legend_pos,textattrs=[text.size(legend_text_size)]),\
       x = graph.axis.linear(min=-2,max=-1,title = None,manualticks=mymanualticks),\
       y2 = graph.axis.linear(min=np.min(value_range)-0.5,max=np.max(value_range)+0.5,manualticks = mymanualticks_y,title = '$a_R$',parter=None)))

# "Paint" the plot:
min_val = np.max(1./(a_log))
max_val = np.min(1./(a_log))
m = 1./(max_val-min_val)
n = -min_val*m

for i in range(len(value_range)):
   the_color = color.cmyk(1.-(m*(1./(tick_range[i]))+n),m*(1./(tick_range[i]))+n,m*(1./(tick_range[i]))+n,0)
   draw_rectangular_band(cbar, -2., value_range[i] - 0.5, 1., 1.0, the_color)

# Plot borders of cb
cbar = c.insert(graph.graphxy(height=pheight*1.5,width=pwidth/4., xpos = g_i_tpl.xpos + 1.35*delta_x, \
                                                                  ypos = g_i_tpl.ypos+g_i_tpl.height+2.*delta_y,\
       key=graph.key.key(pos=legend_pos,textattrs=[text.size(legend_text_size)]),\
       x = graph.axis.linear(min=-2,max=-1,title = None,manualticks=mymanualticks),\
       y2 = graph.axis.linear(min=np.min(value_range)-0.5,max=np.max(value_range)+0.5,manualticks = mymanualticks_y2,title = None,parter=None)))
"""
###########################################################################################

################# SYMBOL LEGEND EMULATION #################################################
# Define plot that will emulate the colorbar. First, define a fake x-axis:
mymanualticks = [graph.axis.tick.tick(-2,label=""),graph.axis.tick.tick(-1,label="")]

# Now a true y-axis with values I want to show:
#value_range = [0,1,2]
tick_range = p_show
value_range = range(len(p_show))
mymanualticks_y = []
mymanualticks_y2 = []
for i in range(len(value_range)):
        mymanualticks_y.append(graph.axis.tick.tick(value_range[i],label=str(tick_range[i])))
        mymanualticks_y2.append(graph.axis.tick.tick(value_range[i],label=""))

sbar = c.insert(graph.graphxy(height=pheight,width=pwidth/4., xpos = g_a_tpl.xpos + 1.35*delta_x, \
                                                              ypos = g_a_tpl.ypos+g_a_tpl.height-2*delta_y,\
       key=graph.key.key(pos=legend_pos,textattrs=[text.size(legend_text_size)]),\
       x = graph.axis.linear(min=-2,max=-1,title = None,manualticks=mymanualticks),\
       y2 = graph.axis.linear(min=np.min(value_range)-0.5,max=np.max(value_range)+0.5,manualticks = mymanualticks_y,title = '$p$',parter=None)))

# "Paint" the plot:
for i in range(len(value_range)):
   the_size = p_scale/np.log10(p_scale2*tick_range[i])
   the_color = color.cmyk.Black

   # Now plot:
   sbar.plot(graph.data.values(x=[-1.5], y=[value_range[i]], title = None),\
            styles = [graph.style.symbol(graph.style.symbol.circle, symbolattrs = [deco.filled([the_color]), deco.stroked([the_color])],size = the_size)])
##########################################################################################


if fit_option == "float":
    draw_text(sbar, -1.5, np.max(value_range)+1, '$b='+str(b)+'$',text_size=0)
    draw_text(g_p_lin, min_x + (max_x-min_x)/2., max_y_p + ((np.abs(max_y_p-min_y_p))/(np.abs(max_y_p_2-min_y_p_2)))*.2, 'Linear law',text_size=0)
    draw_text(g_p_quad, min_x + (max_x-min_x)/2., max_y_p_2 + .2, 'Quadratic law',text_size=0)
    draw_text(g_p_log, min_x + (max_x-min_x)/2., max_y_p_2 + .2, 'Logarithmic law',text_size=0)
    draw_text(g_p_sqrt, min_x + (max_x-min_x)/2., max_y_p_2 + .2, 'Square-root law',text_size=0)
    draw_text(g_p_tpl, min_x + (max_x-min_x)/2., max_y_p_2 + .2, 'Three-parameter law',text_size=0)
else:
    draw_text(sbar, -1.5, np.max(value_range)+1, '$b='+str(b)+'$',text_size=0)
    draw_text(g_p_quad, min_x + (max_x-min_x)/2., max_y_p + .3, 'Quadratic law (fixed LDs fit)', text_size=0)
    draw_text(g_p_log, min_x + (max_x-min_x)/2., max_y_p + .3, 'Logarithmic law (fixed LDs fit)',text_size=0)
    draw_text(g_p_sqrt, min_x + (max_x-min_x)/2., max_y_p + .3, 'Square-root law (fixed LDs fit)',text_size=0)
    draw_text(g_p_tpl, min_x + (max_x-min_x)/2., max_y_p + .3, 'Three-parameter law (fixed LDs fit)',text_size=0)

# Now plot each point at a time, where p defines the size and the color defines the inclination.
# Do this for each LD law differently:
for method in ['lin','log','sqrt','quad','tpl']:
        print '\t ################################'
        print '\t RESULTS FOR '+method+' LD law'
        print '\t ################################'
        exec 'c_idx = np.argsort(p_'+method+')[::-1]'
        for variable in variables:
                exec variable+method+'='+variable+method+'[c_idx]'
        # Calculate maximum and minimum bias for each of the fitted values
        # and each of the different methods:
        exec 'teff = input_teff_'+method
        exec 'p = p_'+method
        exec 'a = a_'+method
        exec 'inclinations = (np.arccos(b/a_'+method+'))*(180./np.pi)'
        # Estimate biases on p, a and i: 
        exec 'p_bias = (p - p_'+fit_option+'_'+method+')/(p)'
        exec 'a_bias = (a - a_'+fit_option+'_'+method+')/(a)'
        #exec 'i_bias = (inclinations - i_'+fit_option+'_'+method+')/(inclinations)'
        # Estimate maximum and minimum bias on p:
        idx_max_p = np.where(p_bias == np.max(p_bias))[0]
        idx_min_p = np.where(p_bias == np.min(p_bias))[0]
        print '\t -------- Experiment with '+fit_option+' LDs --------'
        print '\t Biases on p:'
        print '\t >> Maximum positive bias: ',np.max(p_bias[idx_max_p])*100,'%'
        print '\t >> At Teff = ',teff[idx_max_p],'(grid number '+str(idx_max_p)+')'
        print '\t >> Maximum negative bias: ',np.max(p_bias[idx_min_p])*100,'%'
        print '\t >> At Teff = ',teff[idx_min_p],'(grid number '+str(idx_min_p)+')\n'
        # Same for a:
        idx_max_a = np.where(a_bias == np.max(a_bias))[0]
        idx_min_a = np.where(a_bias == np.min(a_bias))[0]
        print '\t Biases on a:'
        print '\t >> Maximum positive bias: ',np.max(a_bias[idx_max_a])*100,'%'
        print '\t >> At Teff = ',teff[idx_max_a],'(grid number '+str(idx_max_a)+')'
        print '\t >> Maximum negative bias: ',np.max(a_bias[idx_min_a])*100,'%'
        print '\t >> At Teff = ',teff[idx_min_a],'(grid number '+str(idx_min_a)+')\n'
        # And for i:
        #idx_max_i = np.where(i_bias == np.max(i_bias))[0]
        #idx_min_i = np.where(i_bias == np.min(i_bias))[0]
        #print '\t Biases on i:'
        #print '\t >> Maximum positive bias: ',np.max(i_bias[idx_max_i])*100,'%'
        #print '\t >> At Teff = ',teff[idx_max_i],'(grid number '+str(idx_max_i)+')'
        #print '\t >> Maximum negative bias: ',np.max(i_bias[idx_min_i])*100,'%'
        #print '\t >> At Teff = ',teff[idx_min_i],'(grid number '+str(idx_min_i)+')\n'

        # Now, for plotting, calculate minimum and maximum values of a:
        exec 'min_val = np.max(1./(a_'+method+'))'
        exec 'max_val = np.min(1./(a_'+method+'))'
        m = 1./(max_val-min_val)
        n = -min_val*m

        exec 'g_p = g_p_'+method
        exec 'g_a = g_a_'+method
        #exec 'g_i = g_i_'+method
        for c_p in p_show[::-1]:
         all_c_p_idx = np.where(c_p == p)[0]
         for i in all_c_p_idx[np.argsort(a[all_c_p_idx])]:
        #for i in range(len(p)):
        #   if p[i] in p_show:#>=p_min and p[i]<=p_max:
            # Define the color of the point. For reference: Red (0,1,1,0) to Cyan (1,0,0,0)
            #print 'teff:',teff[i],'a_R:',a[i]
            #print '(',1.-(m*np.log10(a[i])+n),',',m*np.log10(a[i])+n,',',m*np.log10(a[i])+n,',0)'
            the_color = color.cmyk.Grey#color.cmyk(1.-(m*(1./(a[i]))+n),m*(1./(a[i]))+n,m*(1./(a[i]))+n,0)
            # Define the size of the point:
            the_size = p_scale/np.log10(p_scale2*p[i])

            # Now plot:
            g_p.plot(graph.data.values(x=[teff[i]], y=[-100*(p_bias[i])], title = None),\
            styles = [graph.style.symbol(graph.style.symbol.circle, symbolattrs = [deco.filled([the_color]), deco.stroked([color.cmyk.Black,style.linewidth.THIN])],size = the_size)])

            g_a.plot(graph.data.values(x=[teff[i]], y=[-100*(a_bias[i])], title = None),\
            styles = [graph.style.symbol(graph.style.symbol.circle, symbolattrs = [deco.filled([the_color]), deco.stroked([color.cmyk.Black,style.linewidth.THIN])],size = the_size)])

            #g_i.plot(graph.data.values(x=[teff[i]], y=[-100*(i_bias[i])], title = None),\
            #styles = [graph.style.symbol(graph.style.symbol.circle, symbolattrs = [deco.filled([the_color]), deco.stroked([color.cmyk.Black,style.linewidth.THIN])],size = the_size)])

        # Finally, plot the lines that mark zero bias:

        g_p.plot(graph.data.values(x=[np.min(teff),np.max(teff)], y=[0,0], title = None),\
                                      styles = [graph.style.line([color.cmyk.Black,\
                                      style.linestyle.dashed,\
                                      style.linewidth.thin])])

        g_a.plot(graph.data.values(x=[np.min(teff),np.max(teff)], y=[0,0], title = None),\
                                      styles = [graph.style.line([color.cmyk.Black,\
                                      style.linestyle.dashed,\
                                      style.linewidth.thin])])

        #g_i.plot(graph.data.values(x=[np.min(teff),np.max(teff)], y=[0,0], title = None),\
        #                              styles = [graph.style.line([color.cmyk.Black,\
        #                              style.linestyle.dashed,\
        #                              style.linewidth.thin])])


c.writeEPSfile(outname,write_mesh_as_bitmap = True,write_mesh_as_bitmap_resolution=2)
c.writePDFfile(outname,write_mesh_as_bitmap = True,write_mesh_as_bitmap_resolution=2)
