#
# https://doi.org/10.1016/j.geoderma.2021.115332
# https://github.com/AlexandreWadoux/MapQualityEvaluation
#

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.floating_axes as fa

def SolarDiagram(mods, obs, modnames=None, fig=None):

    rs = [0.95, 0.9, 0.7, 0]
    x_axis_begin=-1
    x_axis_end=1
    y_axis_end=1.4
    by=0.2

    def compute_model_nME(x, y):
        nme = (np.mean(x) - np.mean(y)) / np.std(y)
        return nme

    def NSTD(pred, expd):                                                                                 # normalized standard deviation
        return np.std(pred) / np.std(expd)

    def compute_model_uRMSDnorm(x, y):
        #sg = np.std(mod) / np.std(obs)
        #model_uRMSDnorm = [np.sqrt(1 + sg**2 - 2*sg*pearsonr(mod, obs)[0]) for mod in mods]
        sigma_ast = np.std(x) / np.std(y)
        corxy = np.corrcoef(x, y)[0, 1]
        return np.sqrt(1 + sigma_ast*(sigma_ast - 2*corxy))

    def compute_sigmaD(x, y):
        return np.sign(np.std(x) - np.std(y))

    def RMSDfromR1(R1):
        #return np.sqrt(1+R1**2-2*R1**2)
        return np.sqrt(1+R1**2-2*R1**2)

    model_nME       = [compute_model_nME(x, obs) for x in mods]
    model_uRMSDnorm = [compute_model_uRMSDnorm(x, obs) for x in mods]       # * compute_sigmaD(x, obs)

    circles = [{
        'x': RMSDfromR1(r)*np.cos(np.linspace(0, np.pi, 300)),
        'y': RMSDfromR1(r)*np.sin(np.linspace(0, np.pi, 300)),
        'label': f'{r}'
    } for r in rs]

    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot()

    for circle in circles:
        ax.plot(circle['x'], circle['y'], '--', color='black', linewidth=0.8)
        ax.text(circle['x'][100]+0.01, circle['y'][100]+0.01, circle['label'], color='DarkGrey')        #, transform=ax.transAxes, rotation=270.+alpha, fontdict=self.fontax, color=self.gray)
    ax.scatter(model_nME, model_uRMSDnorm, c='Pink', s=30, edgecolor='Black')                           #c=colorval, cmap='viridis')

    ax.set_xlabel('ME*')
    ax.set_xlim([x_axis_begin-0.005, x_axis_end+0.005])
    ax.set_xticks(np.arange(x_axis_begin, x_axis_end+by, by))

    ax.set_ylabel('SDE*')
    ax.set_ylim([0, y_axis_end])
    ax.set_yticks(np.arange(0, y_axis_end+by, by))

    if modnames:
        for i, txt in enumerate(modnames):
            ax.annotate(txt, (model_nME[i], model_uRMSDnorm[i]), textcoords="offset points", ha='center', xytext=(0, 7))

    ## Add legend
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return fig


if '__main__'== __name__:
    # Example usage
    obs = np.array([1, 2, 5, 6, 4, 3])
    mo1 = np.array([1, 2, 6, 5, 4, 3])
    mo2 = np.array([2, 3, 4, 8, 6, 2])
    mo3 = np.array([0, 1, 4, 5, 6, 2])
    mods = [mo1, mo2, mo3]

    '''from scipy import stats
    for mod in mods:
        rp = stats.pearsonr(obs, mod)[0]
        rn = np.corrcoef(obs, mod)[0, 1]
        obsm = np.mean(obs)
        modm = np.mean(mod)
        rm = np.sum((obs-obsm) * (mod-modm)) / (np.std(obs)*np.std(mod)) / len(obs)
        print(rp, rn, rm)
    exit()'''

    fig = plt.figure( figsize = (9,6) )
    plt.style.use('ggplot')
    SolarDiagram(mods, obs, modnames=['Good','Bad','NotSo'])
    plt.savefig('SolarDiagram.png', format='png', bbox_inches='tight', pad_inches=0.1, dpi=200)
