"""
This file performs the extrapolation analysis for the gA results using MDWF on HISQ ensembles
Required Python libraries/software
    numpy
    scipy
    matplotlib (assume latex is used to render fonts)
    pytables (tables)
    iminuit
"""
import os, sys
import argparse, traceback

try:
    import numpy as np
    np.set_printoptions(linewidth=180)
    import scipy as sp
    import scipy.linalg as spla
    import scipy.special as spsp
    import matplotlib.pyplot as plt
    import tables as h5
    import iminuit as mn
    import sqlite_store as sql
    import tqdm
    # modules created for this work
    import ga_fit_funcs as gafit
    import utils
    import data_params as dp
except ImportError as e:
    print type(e)
    print e
    exit()

def parse_input():
    parser = argparse.ArgumentParser(description='''
    perform extrapolation analysis of c51 gA results
    Required Python Libraries/Software
      numpy v1.10.2+
      scipy
      matplotlib
      tables (hdf5)
      iminuit
      sqlite3''',formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('-p','--plot',default=True,action='store_false',\
        help='plot extrapolations? [%(default)s]')
    parser.add_argument('-f','--fits',default='all',action='store',\
        help='''what type of extrapolation to perform? [%(default)s]
        all [t_a2, t_esq_1_a0, t_esq_1_a2, x_nlo_a2, x_nlo_aSa2]
        other [xma_nlo_a2, t_e_1_a2, t_e_2_a2, t_esq_2_a2, x_nnlo_a2 ]
        ''')
    parser.add_argument('--file',type=str,default='c51_gA_mdwf.h5',action='store',
        help='hdf5 input file name [%(default)s]')
    parser.add_argument('--e0',type=float,default=0.,action='store',\
        help='add e_pi offset (e0) for Taylor expansion fits [%(default)s]')
    parser.add_argument('--bs',default=False,action='store_true',\
        help='loop over bootstraps? [%(default)s]')
    parser.add_argument('--error_x',default=True,action='store_false',\
        help='include error in all "x" parameters in analysis? [%(default)s]')
    parser.add_argument('--error_epi',default=True,action='store_false',\
        help='include error in "epi" parameters in analysis? [%(default)s]')
    parser.add_argument('--error_a',default=True,action='store_false',\
        help='include error in "a" parameters in analysis? [%(default)s]')
    parser.add_argument('--error_mL',default=True,action='store_false',\
        help='include error in "mL" parameters in analysis? [%(default)s]')
    parser.add_argument('--error_pq',default=True,action='store_false',\
        help='include error in "PQ" parameters in analysis? [%(default)s]')
    parser.add_argument('--error_mju',default=True,action='store_false',\
        help='include error in "mju" parameters in analysis? [%(default)s]')
    parser.add_argument('--Nbs',type=int,action='store',\
        help='How many bootstrap samples? [default=All]')
    parser.add_argument('--g0fv',nargs=2,type=float,action='store',\
        help='add prior and width to NLO FV coefficient')
    parser.add_argument('--g0b',nargs=2,type=float,action='store',\
        help='add prior and width to MA axial coupling, \bar{g}_0')
    parser.add_argument('--show_fv',default=False,action='store_true',\
        help='show raw results? [%(default)s]')
    parser.add_argument('--epi_x',nargs=2,type=float,action='store',default=[0.,0.27],\
        help='chose x-range for gA vs e_pi plot [%(default)s]')
    parser.add_argument('--epi_y',nargs=2,type=float,action='store',default=[1.0,1.44],\
        help='chose y-range for gA vs e_pi plot [%(default)s]')
    parser.add_argument('--asq_x',nargs=2,type=float,action='store',default=[-0.01,0.81],\
        help='chose x-range for gA vs e_pi plot [%(default)s]')
    parser.add_argument('--asq_y',nargs=2,type=float,action='store',default=[1.0,1.44],\
        help='chose y-range for gA vs e_pi plot [%(default)s]')
    args = parser.parse_args()
    print('Arguments passed')
    print args
    print('')
    return args

def ini_vals(select):
    # initial values for minimizer
    if select in ['t_esq_1_a2','t_esq_1_aSa2']:
        return {'c0':1.25,'error_c0':0.05,'cm1':-1,'error_cm1':0.05,\
                'ca2':-0.1,'error_ca2':0.02,'g0fv':1.5,'error_g0fv':0.1}
    elif select in ['x_nlo_a2','x_nlo_aSa2']:
        return {'g0':1.25,'error_g0':0.05,'c2':-1,'error_c2':0.05,\
                'ca2':-0.1,'error_ca2':0.02}
    elif select in ['t_a2']:
        return {'c0':1.25,'error_c0':0.05,\
                'ca2':-0.1,'error_ca2':0.02,'g0fv':1.5,'error_g0fv':0.1}
    elif select in ['t_esq_1_a0']:
        return {'c0':1.25,'error_c0':0.05,'cm1':-1,'error_cm1':0.05,\
                'g0fv':1.5,'error_g0fv':0.1}
    elif select in ['xma_nlo_a2']:
        return {'g0':1.25,'error_g0':0.05,'c2':-5,'error_c2':0.05,\
                'g0b':0.5,'error_g0b':0.05,'ca2':-0.1,'error_ca2':0.02}
    else:
        print('initial value is undefined')
        raise SystemExit


if __name__=='__main__':
    # parse keyboard inputs
    args = parse_input()
    # set chipt parameters
    params_gA = dp.gA_parameters()
    # set plotting parameters
    params_plot = dp.plotting_parameters()
    # read data
    data = utils.read_data(args.file,args,params_gA)
    # fit data
    rdict = gafit.fit_gA(args,params_gA,data,ini_vals)
    # plot result
    plot = utils.plot_fit(args,params_gA,params_plot,data,rdict)


    
