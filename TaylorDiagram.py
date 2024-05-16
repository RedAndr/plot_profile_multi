#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified Taylor diagram (Taylor, 2001) test implementation.
see 

Created on Fri Dec 05 15:39:18 2014

@author: laurence
inspired by:
"Yannick Copin's <yannick.copin@laposte.net> Taylor Diagram implimentation
  https://gist.github.com/ycopin/3342888"
  
Sean Elvidge's European Space Weather Week 11 (2014) talk 
  http://www.stce.be/esww11/contributions/public/splinters/metrics/SeanElvidge/

Elvidge, S., M. J. Angling, and B. Nava (2014), 
  On the use of modified Taylor diagrams to compare ionospheric assimilation models, 
  Radio Sci., 49, 737–745, doi:10.1002/2014RS005435.
  
Taylor, K.E. (2001):  
  Summarizing multiple aspects of model performance in a single diagram. 
  J. Geophys. Res., 106, 7183-7192 
  (also see PCMDI Report 55, http://www-pcmdi.llnl.gov/publications/ab55.html)

see also
http://www-pcmdi.llnl.gov/about/staff/Taylor/CV/Taylor_diagram_primer.htm  

Modified BSD License

Copyright (c) 2014 Laurence Billingham.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the Scikit-learn Developers  nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission. 


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.

Link: https://bitbucket.org/lbillingham/modifiedtaylordiagrams/src/master/
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as fa
import mpl_toolkits.axisartist.grid_finder as gf
from scipy.optimize import minimize


class TaylorDiagramPoint(object):
  """
  A single point on a Modified Taylor Diagram.
  How well do the values predicted match the values expected

      * do the means match
      * do the standard deviations match
      * are they correlated
      * what is the normalized error standard deviation
      * what is the bias?

  Notation:

      * s_ := sample standard deviation
      * m_ := sample mean
      * nesd := normalized error standard deviation;
                > nesd**2 = s_predicted**2 + s_expected**2 -
                            2 * s_predicted * s_expected * corcoeff
  """
  def __init__(self, expected, predicted, pred_name, point_id):

      self.pred     = predicted
      self.expd     = expected
      self.s_pred   = np.std(self.pred)
      self.s_expd   = np.std(self.expd)
      self.s_normd  = self.s_pred / self.s_expd
      self.bias     = (np.mean(self.pred) - np.mean(self.expd)) / self.s_expd
      self.corrcoef = np.corrcoef(self.pred, self.expd)[0, 1]
      self.corrcoef = min([self.corrcoef, 1.0])
      self.nesd     = np.sqrt(self.s_pred**2 + self.s_expd**2 - 2 * self.s_pred * self.s_expd * self.corrcoef)
      self.name     = pred_name
      self.point_id = point_id


class ModTaylorDiagram(object):
  """
    Given predictions and expected numerical data
    plot the standard deviation of the differences and correlation between
    expected and predicted in a single-quadrant polar plot, with
    r=stddev and theta=arccos(correlation).
  """
  def __init__(self, fig=None, label='expected'):
      """
      Set up Taylor diagram axes.
      """

      self.title_polar    = 'Pearson correlation coefficient'                        #r'Correlation'
      self.title_xy       = r'Normalized Standard Deviation'
      self.title_expected = r'Expected'
      self.fontax         = {'family':'monospace', 'weight':'semibold', 'size':13 }

      self.max_normed_std = 1.8
      self.s_min          = 0.0
      self.label          = label
      self.gray           = '#404040'

      # Correlation labels
      corln_s    = np.array(['0','0.2','0.4','0.6','0.7','0.8','0.9','0.95','0.99','1'])                                        # List of text labels
      corln_r    = corln_s.astype(np.float64)                                                                                     # Convert them to float
      corln_ang  = np.arccos(corln_r)                                                                                           # Conversion to polar angles
      grid_loc1  = gf.FixedLocator(corln_ang)                                                                                   # Positions

      # Normalized Standard Deviation labels
      nsd_s      = np.array(['0','0.2','0.4','0.6','0.8','1','1.2','1.4','1.6'])                                                # text
      nsd_r      = nsd_s.astype(np.float64)                                                                                     # float
      grid_loc2  = gf.FixedLocator(nsd_r)                                                                                       # positions

      # Normalized standard deviation axis
      tr = PolarAxes.PolarTransform()
      grid_helper = fa.GridHelperCurveLinear( tr,
                                    extremes=(0, np.pi/2.0,                        # 1st quadrant
                                    self.s_min, self.max_normed_std),
                                    grid_locator1   = grid_loc1,
                                    grid_locator2   = grid_loc2,
                                    tick_formatter1 = gf.DictFormatter(dict(zip(corln_ang, corln_s))),
                                    tick_formatter2 = gf.DictFormatter(dict(zip(    nsd_r, nsd_s  )))  )
      self.fig = fig
      if self.fig is None:
          self.fig = plt.figure()

      plt.ioff()
      mpl.use('Agg')
      mpl.pyplot.style.use('ggplot')                          # Plotting style for matplotlib

      # setup axes
      ax = fa.FloatingSubplot(self.fig, 111, grid_helper=grid_helper)
      # make the axis (polar ax child used for plotting)
      self.ax = self.fig.add_subplot(ax)
      # hide base-axis labels etc
      self.ax.axis['bottom'].set_visible(False)
      self._setup_axes()

      # attach the polar axes
      self.polar_ax = self.ax.get_aux_axes(tr)

      self.points = []

      self.plot_pt_id = 1


  def add_prediction(self, expected, predicted, predictor_name, plot_pt_id=None):
      """
      Add a prediction/model to the diagram
      """
      if plot_pt_id == None:
        plot_pt_id = f'{self.plot_pt_id}'
        self.plot_pt_id += 1
      this_point = TaylorDiagramPoint(expected, predicted, predictor_name, plot_pt_id)
      self.points.append(this_point)
      return this_point.s_normd, this_point.corrcoef, this_point.nesd, this_point.bias

  def plot(self):
      "Place all the loaded points onto the figure"

      self._plot_req1_cont(self.label)                                                # Add norm error stddev and nesd==1 contours
      self._plot_nesd_cont(levels=np.arange(0.0, 2.0, 0.2))

      rs         = []
      thetas     = []
      biases     = []
      names      = []
      point_tags = []
      for point in self.points:
          rs        .append(point.s_normd)                                #s_expd
          thetas    .append(np.arccos(point.corrcoef))
          biases    .append(point.bias)
          names     .append(point.name)
          point_tags.append(point.point_id)

      thetas = np.array(thetas)
      rs     = np.array(rs    )
      sc     = self.polar_ax.scatter(thetas, rs, c=biases, s=30, cmap=plt.cm.bwr, vmin=-.5, vmax=.5, edgecolor='Black')
      xy     = self.polar_ax.transData.transform(np.c_[thetas,rs])

      def distfunc(xys):
          s = 0.0
          for xy1 in xys:
              for xy2 in xys:
                  s += np.sqrt((xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2)
          return s

      def ang2xy(angs):                                                                                 # convert angles to xy with distance d
          d = 30
          xyr=[]
          for ang,x_y in zip(angs,xy):
              xyr += [[x_y[0]+d*np.cos(ang), x_y[1]+d*np.sin(ang)]]
          return np.array(xyr)

      def funcang(angs):                                                                                # inverse function - for minimization
          return -distfunc( ang2xy( angs ) )

      angs0 = np.arctan2(xy[:,1]-xy[:,1].mean(), xy[:,0]-xy[:,0].mean())                                # initial distribution - from center to outside
      resx = minimize(funcang, angs0).x   #, method='BFGS'                                              # optimized distribution - as far as possible from each other
      gxy = ang2xy(resx)                                                                                # convert to xy

      for i, tag in enumerate(point_tags):
          x2 = gxy[i,0]-xy[i,0]; y2 = gxy[i,1]-xy[i,1]
          self.polar_ax.annotate(tag, xy=(thetas[i],rs[i]), xycoords='data', xytext=(x2,y2),
                                 textcoords='offset points', arrowprops=dict(arrowstyle='->', color='Black'), size=8 )

      self.fig.subplots_adjust(left=0.05,right=0.6,top=0.8)
      cbaxes = self.fig.add_axes([0.238, 0.87, 0.55, 0.03])
      cbartik = [-.5,-.4,-.3,-.2,-.1,0,.1,.2,.3,.4,.5]
      cbar = plt.colorbar(sc, cax=cbaxes, orientation='horizontal',ticks=cbartik)
      cbarlab = ['%.1f'%t for t in cbartik];  cbarlab[5] = '0'
      cbar.ax.set_xticklabels(cbarlab)
      cbaxes.set_xlabel('Normalized bias', fontdict=self.fontax)
      cbaxes.xaxis.set_ticks_position('top')
      cbaxes.xaxis.set_label_position('top')
      self.show_key()


  def show_key(self):
      "add annotation key for model IDs and normalization factors"
      textstr1 = 'N  Model       NormStDev Correl   NormErr NormBias'
      textstr2 = ''
      for i, p in enumerate(self.points):
          textstr2 += '{0:2s} {1:10s}   {2:.4f}   {3:.4f}   {4:.4f}   {5: .4f}\n'.format(p.point_id[:2], p.name[:10], p.s_normd, p.corrcoef, p.nesd, p.bias)
      textstr2 = textstr2[:-1]
      #props = dict(boxstyle='round', facecolor='white', alpha=0.7)
      props1 = dict(facecolor='#E0E0E0')
      props2 = dict(facecolor='#F0F0F0')
      # place a text box in upper left in axes coords
      font = {'family':'monospace', 'weight':'normal', 'size':9}
      self.ax.text(1, 1.05, textstr1, transform=self.ax.transAxes, verticalalignment='top', bbox=props1, fontdict=font)
      self.ax.text(1, 1.00, textstr2, transform=self.ax.transAxes, verticalalignment='top', bbox=props2, fontdict=font)

  def show_norm_factor(self):
      "add annotation about the normalization factor"
      n_fact = self.points[0]
      out_str = r'Norm Factor {:.2f}'.format(n_fact)
      x = 0.95 * self.max_normed_std
      y = 0.95 * self.max_normed_std
      self.ax.text(x, y,
                  out_str,
                  horizontalalignment='right',
                  verticalalignment='top',
                  bbox={'edgecolor': 'black', 'facecolor':'None'})

  def _plot_req1_cont(self, label):
      "plot the normalized standard deviation = 1 contour and label"
      t = np.linspace(0, np.pi/2)
      r = np.ones_like(t)
      self.polar_ax.plot(  t,   r, '--', color=self.gray, label=label)
      self.polar_ax.plot([0], [1], 'o' , color=self.gray, label=label, mec='Black')

  def _plot_nesd_cont(self, levels=8):
      "plot the normalized error standard deviation contours"
      rs, ts = np.meshgrid(np.linspace(self.s_min, self.max_normed_std, num=64), np.linspace(0, np.pi/2, num=64))
      nesd = np.sqrt(1.0 + rs**2 - 2 * rs * np.cos(ts))
      contours = self.polar_ax.contour(ts, rs, nesd, levels, linestyles='dotted', linewidths=1.0, colors='DarkGray')

  def _setup_angle_axis(self):
      "set the ticks labels etc for the angle axis"
      loc = 'top'
      self.ax.axis[loc].set_axis_direction('bottom')
      self.ax.axis[loc].toggle(ticklabels=True, label=True)
      self.ax.axis[loc].major_ticklabels.set_axis_direction('top')
      self.ax.axis[loc].label.set_axis_direction('top')
      gip = 1.04                # hipotenuse length
      stp = 1.4                 # step in degree
      mid = 43.0                # middle of the title in degree
      lst = len(self.title_polar)/2.*stp
      sta = mid+lst
      end = mid-lst-stp
      for alpha,char in zip(np.arange(sta,end,-stp),self.title_polar[:]):
          ixe = gip * np.cos(alpha/180.*np.pi)
          ygr = gip * np.sin(alpha/180.*np.pi)
          self.ax.text(ixe, ygr, char, transform=self.ax.transAxes, rotation=270.+alpha, fontdict=self.fontax, color=self.gray)

  def _setup_x_axis(self):
      "set the ticks labels etc for the x axis"
      loc = 'left'
      self.ax.axis[loc].set_axis_direction('bottom')
      self.ax.axis[loc].label.set_family(self.fontax['family'])
      self.ax.axis[loc].label.set_weight(self.fontax['weight'])
      self.ax.axis[loc].label.set_size  (self.fontax['size']  )
      self.ax.axis[loc].label.set_text  (self.title_xy)

  def _setup_y_axis(self):
      "set the ticks labels etc for the y axis"
      loc = 'right'
      self.ax.axis[loc].set_axis_direction('top')
      self.ax.axis[loc].toggle(ticklabels=True)
      self.ax.axis[loc].major_ticklabels.set_axis_direction('left')
      self.ax.axis[loc].label.set_family(self.fontax['family'])
      self.ax.axis[loc].label.set_weight(self.fontax['weight'])
      self.ax.axis[loc].label.set_size  (self.fontax['size']  )
      self.ax.axis[loc].label.set_text(self.title_xy)

  def _setup_axes(self):
      "set the ticks labels etc for the angle x and y axes"
      self._setup_angle_axis()
      self._setup_x_axis()
      self._setup_y_axis()


if '__main__'== __name__:
    fig = plt.figure( figsize = (9,6) )
    plt.style.use('ggplot')

    mtd = ModTaylorDiagram(fig)
    x = np.linspace(0.0, 4.0*np.pi, 100)
    obs = np.sin(x)
    # Models
    pred_0 = obs + 0.2*np.random.randn(len(x))
    pred_1 = 0.8*obs + 2*np.random.randn(len(x))
    pred_2 = np.sin(x - np.pi/10)  - 0.5*np.random.randn(len(x))
    pred_3 = - 0.1*np.random.randn(len(x))
    mods = [pred_0, pred_1, pred_2, pred_3]
    mod_names = [r'Model 0', r'Model 1', r'Model 2', r'Model 3']
    mod_ids = [r'a', r'$\beta$', r'$\spadesuit$', r'f']

    for i, mod in enumerate(mods):
        mtd.add_prediction(obs, mod, mod_names[i], mod_ids[i])

    mtd.plot()
    fig.savefig('TylorDiagram.png',dpi=200)
