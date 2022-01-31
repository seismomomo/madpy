import matplotlib
params = {
    'text.latex.preamble': r'\usepackage{bm} \usepackage{amsmath}',
    'axes.facecolor': 'w',
    'axes.labelsize': 14, 
    'axes.linewidth': 1.25,
    'axes.titlesize': 16,
    'errorbar.capsize': 6,
    'font.family': 'sans-serif',
    'font.size': 14, 
    'image.cmap': 'gray',
    'image.interpolation': 'nearest',
    'image.origin': 'lower',
    'legend.edgecolor': 'k',
    'legend.fancybox': True,
    'legend.fontsize': 12, 
    'legend.framealpha': 0.7,
    'savefig.facecolor': 'w',
    'savefig.dpi': 300,
    'xtick.labelsize': 12,
    'xtick.major.width' : 1,
    'xtick.minor.width' : 0.5,
    'xtick.major.size' : 8,
    'xtick.minor.size' : 4,
    'ytick.labelsize': 12,
    'ytick.major.width' : 1,
    'ytick.minor.width': 0.5,
    'ytick.major.size': 8,
    'ytick.minor.size': 4
}
matplotlib.rcParams.update(params)


def amplitude_plot_parameters():
    """Amplitude plot parameters"""
    
    colors = {
        'dat': 'dimgray',
        'pha': 'cornflowerblue',
        'p2pl': 'forestgreen',
        'p2pm': 'mediumvioletred'
    }
    
    linewidths = {
        'dat': 1,
        'pha': 2,
        'p2pl': 2
    }
    
    labels = {
        'dat': 'data',
        'o': 'o-arr',
        'p': 'p-arr',
        's': 's-arr',
        'p2pm': 'p2p'
    }

    
    return {'c': colors, 'lw': linewidths, 'lab': labels}



def duration_plot_parameters():
    """Duration plot parameters"""
    
    colors = {
        'dat': 'dimgray',
        'pha': 'cornflowerblue',
        'dur': 'darkblue',
        'avg': 'lavender',
        'nth': 'forestgreen',
        'fit': 'mediumvioletred'
    }
    
    linewidths = {
        'dat': 1,
        'pha': 2,
        'dur': 2,
        'avg': 2,
        'nth': 2,
        'fit': 2
    }
    
    labels = {
        'dat': 'data',
        'o': 'o-arr',
        'p': 'p-arr',
        's': 's-arr',
        'dur': 'dur',
        'avg': 'avg',
        'nth': 'thresh',
        'fit': 'linefit'
    }

    
    return {'c': colors, 'lw': linewidths, 'lab': labels}
