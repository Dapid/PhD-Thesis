from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['palatino']})
rc('xtick', labelsize=22)
rc('ytick', labelsize=22)
rc('axes', labelsize=22)
rc('legend', fontsize=18)
rc('axes', titlesize=30)
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb} \usepackage[euler-digits]{eulervm} \usepackage{nicefrac}')

MAROON = '#AD1737'
BLUE = 'RoyalBlue'
GREEN = (0, 0.5, 0)
