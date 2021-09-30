import matplotlib
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'savefig.dpi': 300,  
    'axes.labelsize': 14, 
    'axes.titlesize': 16,
    'axes.linewidth': 1,
    'font.size': 14, 
    'legend.fontsize': 12, 
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.width' : 1,
    'xtick.minor.width' : 0.5,
    'xtick.major.size' : 8,
    'xtick.minor.size' : 4,
    'ytick.major.width' : 1,
    'ytick.minor.width': 0.5,
    'ytick.major.size': 8,
    'ytick.minor.size': 4,
    'font.family': 'sans-serif',
    'errorbar.capsize': 6
    
}
matplotlib.rcParams.update(params)
