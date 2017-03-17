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
    import ga_fit_funcs as gafit
    import sqlite_store as sql
    import tqdm
except ImportError as e:
    print type(e)
    print e
    exit()

def parse_input():
    parser = argparse.ArgumentParser(description='''
    perform extrapolation analysis of c51 gA results
    Required Python Libraries/Software
      numpy
      scipy
      matplotlib
      tables (hdf5)
      iminuit''',formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('-p','--plot',default=True,action='store_false',\
        help='plot extrapolations? [%(default)s]')
    parser.add_argument('-f','--fits',default='all',action='store',\
        help='''what type of extrapolation to perform? [%(default)s]
        all
        taylor_e_1
        taylor_e_2
        taylor_esq_1
        taylor_esq_2
        chiral_nlo
        chiral_nnlo
        chiral_ma_nlo''')
    parser.add_argument('--e0',type=float,default=0.,action='store',\
        help='add e_pi offset (e0) for Taylor expansion fits [%(default)s]')
    parser.add_argument('--bs',default=False,action='store_true',\
        help='loop over bootstraps? [%(default)s]')
    parser.add_argument('--error_epi',default=True,action='store_false',\
        help='include error in "x" parameters in analysis? [%(default)s]')
    parser.add_argument('--Nbs',type=int,action='store',\
        help='How many bootstrap samples? [default=All]')
    parser.add_argument('--g0fv',nargs=2,type=float,action='store',\
        help='add prior and width to NLO FV coefficient')
    parser.add_argument('--g0bar',nargs=2,type=float,action='store',\
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

def chipt_parameters():
    # Physical Point.  We chose the charged pion mass to define the physical
    p = dict()
    p['mpi_phys'] = 139
    p['fpi_phys'] = 130.41
    p['Fpi_phys'] = p['fpi_phys'] / np.sqrt(2)
    p['epi_phys'] = p['mpi_phys'] / (4 * np.pi * p['Fpi_phys'])
    p['ga_phys'] = 1.2723
    p['dga_phys'] = 0.0023
    # define ensembles
    p['ensembles'] = ['a15m310','a12m310', 'a09m310',
                      'a15m220','a12m220S','a12m220','a12m220L',
                      'a15m130']
    p['ens_idx'] = {'a15m310':0,'a12m310':1,'a09m310':2,
                    'a15m220':3,'a12m220S':4,'a12m220':5,'a12m220L':6,
                    'a15m130':7}
    p['l_d'] = len(p['ensembles'])
    # spatial length
    p['e_L']  = {'a15m310':16,'a15m220':24,'a15m130':32,
                 'a12m310':24,'a12m220S':24,'a12m220':32,'a12m220L':40,
                 'a09m310':32}
    # mpi * L
    p['mpiL'] = {'a15m310':3.779,'a15m220':3.973,'a15m130':3.233,
                 'a12m310':4.531,'a12m220S':3.257,'a12m220':4.298,'a12m220L':5.363,
                 'a09m310':4.505}
    # a/w0
    p['aw0']  = {'a15m310':0.8804,'a15m220':0.8804,'a15m130':0.8804,
                 'a12m310':0.7036,'a12m220S':0.7036,'a12m220':0.7036,'a12m220L':0.7036,
                 'a09m310':0.5105}
    # a/w0 uncertainty
    p['daw0'] = {'a15m310':0.0003,'a15m220':0.0003,'a15m130':0.0003,
                 'a12m310':0.0005,'a12m220S':0.0005,'a12m220':0.0005,'a12m220L':0.0005,
                 'a09m310':0.0003}
    # alpha_s
    p['afs']  = {'a15m310':0.58801,'a15m220' :0.58801,'a15m130':0.58801,
                 'a12m310':0.53796,'a12m220S':0.53796,'a12m220':0.53796,'a12m220L':0.53796,
                 'a09m310':0.43356}
    # a/w0 and mL arrays
    xa = np.zeros([p['l_d']])
    mL = np.zeros([p['l_d']])
    for i,ens in enumerate(p['ensembles']):
        xa[i] = p['aw0'][ens]
        mL[i] = p['mpiL'][ens]
    p['xa'] = xa
    p['mL'] = mL
    # sqlite parameters
    p['dbname'] = 'callat_ga.sqlite'
    p['tblname'] = 'xcont'
    return p

def plotting_parameters():
    p = dict()
    # Figure sizes and fonts
    p['fig_gldn'] = (8.125,5.018)
    p['ga_axes'] = [0.095,0.128,0.895,0.865]
    p['mL_axes'] = [0.095,0.125,0.895,0.865]
    p['fs'] = 24

    # set up ensemble parameters for plotting
    # colors
    rd = '#c82506'
    gn = '#70b741'
    bl = '#51a7f9'
    p['cont_color'] = '#b36ae2'
    p['a_cont'] = 0.4
    p['e_clr'] = {'a15m310':rd,'a15m220':rd,'a15m130':rd,
                  'a12m310':gn,'a12m220S':gn,'a12m220':gn,'a12m220L':gn,
                  'a09m310':bl}
    p['m_lbl'] = {'a15m310':r'$m_\pi\sim310$~MeV','a12m310':r'$m_\pi\sim310$~MeV','a09m310':r'$m_\pi\sim310$~MeV',
                  'a15m220':r'$m_\pi\sim220$~MeV','a12m220':r'$m_\pi\sim220$~MeV',
                  'a15m130':r'$m_\pi\sim130$~MeV',
                  'a12m220S':r'$m_\pi\sim220$~MeV','a12m220L':r'$m_\pi\sim220$~MeV'}
    p['m_i'] = ['a15m310','a15m220','a15m130']
    p['a_lbl'] = {'a15m310':r'$a\sim0.15$~fm','a12m310':r'$a\sim0.12$~fm','a09m310':r'$a\sim0.09$~fm',
                  'a15m220':r'$a\sim0.15$~fm','a12m220':r'$a\sim0.12$~fm',
                  'a15m130':r'$a\sim0.15$~fm',
                  'a12m220S':r'$a\sim0.12$~fm','a12m220L':r'$a\sim0.12$~fm'}
    p['a_i'] = ['a15m310','a12m310','a09m310']
    p['e_mrkr'] = {'a15m310':'s','a12m310':'s','a09m310':'s',
                   'a15m220':'h','a12m220':'h','a12m220S':'h','a12m220L':'h',
                   'a15m130':'d'}
    return p

def read_data(args,p):
    data = dict()
    c51_data = h5.open_file('c51_gA_mdwf.h5')
    # check how many Nbs are in file
    if args.Nbs == None:
        Nbs = c51_data.get_node('/gA/'+p['ensembles'][0]+'/bs').read().shape[0]
    else:
        Nbs = args.Nbs
    p['Nbs'] = Nbs
    print('using Nbs = %d samples' %Nbs)

    ga_bs = np.zeros([Nbs,p['l_d']])
    ga_b0 = np.zeros([p['l_d']])
    epi_bs = np.zeros_like(ga_bs)
    epi_b0 = np.zeros_like(ga_b0)
    for i,ens in enumerate(p['ensembles']):
        ga_bs[:,i]  = c51_data.get_node('/gA/'+ens+'/bs').read()[0:Nbs]
        ga_b0[i]    = float(c51_data.get_node('/gA/'+ens+'/b0').read())
        epi_bs[:,i] = c51_data.get_node('/epi/'+ens+'/bs').read()[0:Nbs]
        epi_b0[i]   = float(c51_data.get_node('/epi/'+ens+'/b0').read())
        #print(ens,ga_b0[i],ga_bs.std(axis=0)[i],epi_b0[i],epi_bs.std(axis=0)[i])
    data['ga_bs'] = ga_bs
    data['ga_b0'] = ga_b0
    data['epi_bs'] = epi_bs
    data['epi_b0'] = epi_b0
    c51_data.close()
    return data

def ini_vals(select):
    # initial values for minimizer
    if select in ['taylor_esq_1']:
        return {'c0':1.25,'error_c0':0.05,'cm1':-1,'error_cm1':0.05,\
                'ca2':-0.1,'error_ca2':0.02,'g0fv':1.2,'error_g0fv':0.1}
    else:
        print('initial value is undefined')
        raise SystemExit

class ChiSq():
    def __init__(self,args,p,data):
        self.args = args
        self.p = p
        self.ga_bs = data['ga_bs']
        self.ga_b0 = data['ga_b0']
        self.epi_bs = data['epi_bs']
        self.epi_b0 = data['epi_b0']
        self.do_bs = False
        return None
    def __call__(self,do_bs,bs):
        self.do_bs = do_bs
        self.bs = bs
    def select_chisq(self,select):
        if select in ['taylor_esq_1']:
            return self.taylor_esq_1
        else:
            print('chisq is unselected')
            raise SystemExit
    def taylor_esq_1(self,c0,cm1,ca2,g0fv):
        self.x0 = self.args.e0**2
        self.xphys = self.p['epi_phys']**2
        self.xdict = {'epi0':self.x0, 'epi':self.xphys}
        chisq = 0.
        if self.do_bs:
            y     = self.ga_bs[self.bs]
            x     = (self.epi_bs**2)[self.bs]
        else:
            y     = self.ga_b0
            x     = self.epi_b0**2
        ybs   = self.ga_bs.mean(axis=0)
        xbs   = (self.epi_bs**2).mean(axis=0)
        cdict = {'c0':c0, 'ca2':ca2, 'cm1':cm1}
        f     = gafit.ga_epi(self.x0,x,self.p['xa'],**cdict)
        fbs   = gafit.ga_epi(self.x0,xbs,self.p['xa'],**cdict)
        for i,ens in enumerate(self.p['ensembles']):
            f[i]   += gafit.dgaFV(self.epi_b0[i],self.p['mL'][i],g0fv)
            fbs[i] += gafit.dgaFV(self.epi_b0[i],self.p['mL'][i],g0fv)
        # need to make switch handle error_epi, error_mpi, error_a separately
        if self.args.error_epi:
            cov = np.zeros([self.p['Nbs'],self.p['l_d']])
            for i,ens in enumerate(self.p['ensembles']):
                dy  = self.ga_bs[:,i] - ybs[i]
                df  = gafit.ga_epi(self.x0,(self.epi_bs**2)[:,i],self.p['xa'][i],**cdict) -fbs[i]
                df += gafit.dgaFV(self.epi_bs[:,i],self.p['mL'][i],g0fv)
                cov[:,i]  = ( dy - df )**2
            cov = (1./self.p['Nbs']) * np.sum(cov,axis=0)
        else:
            cov = self.ga_bs.var(axis=0)
        '''
        we have no PQ data on same ensembles so
        y,f,cov are all len(l_d) arrays
        numpy will properly do the multiplication/division
        '''
        chisq += np.sum( (y-f)**2 / cov )
        if self.args.g0fv != None:
            chisq += (g0fv - self.args.g0fv[0])**2 / self.args.g0fv[1]**2
        return chisq

def chipt_fit(args,p,data):
    def print_output(CS,ga_min,select):
        dof = CS.p['l_d'] - len(ga_min.values)
        print "chi^2 = %.4f, dof = %d, Q = %.4f" %(ga_min.fval,dof,gafit.Q(ga_min.fval,dof))
        for i in ga_min.parameters:
            print '  %s = %.4f +- %.4f' %(i,ga_min.values[i],ga_min.errors[i])

        # central value
        x0 = CS.x0
        xphys = CS.xphys
        cov = np.array(ga_min.matrix(correlation=False,skip_fixed=True))[0:-1,0:-1]
        # uncertainty - gA-infinite doesn't know about FV
        # so chop covariance matrix - g0fv is last parameter
        if select in ['taylor_esq_1']:
            params = CS.xdict.copy()
            params.update(ga_min.values)
            ga_fit = gafit.ga_epi(a=0,**params)
            dga_fit = gafit.dga_epi(epi0=x0,epi=np.array([xphys]),a=0,lam_cov=cov,**ga_min.values)
        print('gA = %.3f +- %.3f' %(ga_fit,dga_fit))
        print('g0fv = %.3f +- %.3f' %(ga_min.values['g0fv'],ga_min.errors['g0fv']))
        if args.g0fv != None:
            print('g0fv prior = %f +- %f' %(args.g0fv[0],args.g0fv[1]))
        print('AICc = %.3f\n' %gafit.aicc(ga_min.fval,CS.p['l_d'],len(ga_min.values)))
        return {'ga_fit':ga_fit, 'dga_fit':dga_fit, 'xdict':CS.xdict.copy(), 'ga_min':ga_min}
    # initialized ChiSq class
    CS = ChiSq(args,p,data)
    # collect result
    rdict = dict()
    # choose fit function
    if args.fits in ['all','taylor_esq_1']:
        select = 'taylor_esq_1'
        print('gA = c0 + c1*(epi**2-e0**2) + ca2 * (a/w0)**2\n')
        # do the minimization
        ga_min = gafit.minimize(CS.select_chisq(select),ini_vals(select))
        # print outputs
        rdict[select] = print_output(CS,ga_min,select)
        if args.bs:
            print('make sqlite db and table')
            cur,conn = sql.login(p)
            cur,conn = sql.id_name_nbs_result_table(cur,conn,p)
            ga_min_bs = gafit.minimize(CS.select_chisq(select),ini_vals(select))
            # write boot0
            b0result = ga_min_bs.values
            b0result['cov'] = np.array(ga_min.matrix(correlation=False,skip_fixed=True)).tolist()
            b0result['AICc'] = gafit.aicc(ga_min.fval,CS.p['l_d'],len(ga_min.values))
            b0result['chi2'] = ga_min_bs.fval
            b0result['dof'] = CS.p['l_d'] - len(ga_min.values)
            b0result['Q'] = gafit.Q(ga_min.fval,b0result['dof'])
            b0result = str(b0result).replace("'",'\"')
            cur,conn = sql.id_name_nbs_result_insert(cur,conn,p,select,0,b0result)
            for bs in tqdm.tqdm(range(p['Nbs']),desc='Nbs'):
                CS(True,bs)
                ga_min_bs = gafit.minimize(CS.select_chisq(select),ini_vals(select))
                cur,conn = sql.id_name_nbs_result_insert(cur,conn,p,select,bs+1,str(ga_min_bs.values).replace("'",'\"'))
                raise SystemExit
    return rdict

def plot_fit(args,params_chipt,params_plot,data,rdict):
    ############################
    # FUNCTIONS FOR plot_fit() #
    ############################
    def run_from_ipython():
        try:
            __IPYTHON__
            return True
        except NameError:
            return False
    def continuum_plot(args,params_plot,result,ax,legend,select):
        e0 = result['xdict']['epi0']
        epi = result['xdict']['epi_plot']
        x = result['xdict']['xplot']
        a = result['xdict']['a']
        # taylor fit needs g0fv, which infinite volume function doesn't know about
        # so chop of the last element (g0fv) of corrleation matrix
        cov = np.array(result['ga_min'].matrix(correlation=False,skip_fixed=True))[0:-1,0:-1]
        ga_plot = gafit.ga_epi(epi0=e0,epi=epi,a=a,**result['ga_min'].values)
        if select in ['taylor_esq_1']:
            dga_plot = gafit.dga_epi(epi0=e0,epi=epi,a=a,lam_cov=cov,**result['ga_min'].values)
            if type(epi) != np.ndarray and type(a) == np.ndarray:
                leg, = ax.fill(x,-100*np.ones_like(x),\
                    color=params_plot['cont_color'],alpha=params_plot['a_cont'],\
                    label=r'$g_A^{LQCD}(\epsilon_\pi^{phys},a/w_0)$')
            elif type(a) != np.ndarray and type(epi) == np.ndarray:
                leg, = ax.fill(x,-100*np.ones_like(x),\
                    color=params_plot['cont_color'],alpha=params_plot['a_cont'],\
                    label=r'$g_A^{LQCD}(\epsilon_\pi,a=0)$')
        ax.fill_between(x,ga_plot-dga_plot,ga_plot+dga_plot,\
            color=params_plot['cont_color'],alpha=params_plot['a_cont'])
        legend.append(leg)
        return legend
    def discrete_plot(args,params_plot,data,result,ax,legend,select):
        e0 = result['xdict']['epi0']
        epi = result['xdict']['epi_plot']
        x = result['xdict']['xplot']
        a = result['xdict']['a']
        cov = np.array(result['ga_min'].matrix(correlation=False,skip_fixed=True))[0:-1,0:-1]
        if select in ['taylor_esq_1','taylor_e_1','taylor_esq_2','taylor_e_2']:
            if type(epi) != np.ndarray and type(a) == np.ndarray:
                e = np.array([data['epi_b0'][params_chipt['ens_idx']['a15m310']],\
                    data['epi_b0'][params_chipt['ens_idx']['a15m220']],\
                    data['epi_b0'][params_chipt['ens_idx']['a15m130']]])
                if 'esq' in select:
                    e = e**2
                ga_1 = gafit.ga_epi(epi0=e0,epi=e[0],a=a,**result['ga_min'].values)
                ga_2 = gafit.ga_epi(epi0=e0,epi=e[1],a=a,**result['ga_min'].values)
                ga_3 = gafit.ga_epi(epi0=e0,epi=e[2],a=a,**result['ga_min'].values)
                ga_plot_lbl = [r'$g_A(\epsilon_\pi^{(310)},a/w_0)$',r'$g_A(\epsilon_\pi^{(220)},a/w_0)$',r'$g_A(\epsilon_\pi^{(130)},a/w_0)$']
                color = ['k','k','k']
                ls = ['-.','--','-']
            elif type(a) != np.ndarray and type(epi) == np.ndarray:
                ga_1 = gafit.ga_epi(e0,epi,params_chipt['aw0']['a15m310'],**result['ga_min'].values)
                ga_2 = gafit.ga_epi(e0,epi,params_chipt['aw0']['a12m310'],**result['ga_min'].values)
                ga_3 = gafit.ga_epi(e0,epi,params_chipt['aw0']['a09m310'],**result['ga_min'].values)
                ga_plot_lbl = [r'$g_A(\epsilon_\pi,a=0.15)$',r'$g_A(\epsilon_\pi,a=0.12)$',r'$g_A(\epsilon_\pi,a=0.09)$']
                color = [params_plot['e_clr']['a15m310'],params_plot['e_clr']['a12m310'],params_plot['e_clr']['a09m310']]
                ls = ['-','-','-']
        ga_plot_a = [ga_1,ga_2,ga_3]
        leg, = ax.plot(x,ga_1,color=color[0],alpha=0.5,label=ga_plot_lbl[0],ls=ls[0])
        legend.insert(0,leg)
        leg, = ax.plot(x,ga_2,color=color[1],alpha=0.5,label=ga_plot_lbl[1],ls=ls[1])
        legend.insert(0,leg)
        leg, = ax.plot(x,ga_3,color=color[2],alpha=0.5,label=ga_plot_lbl[2],ls=ls[2])
        legend.insert(0,leg)
        return legend
    def data_plot(args,params_chipt,params_plot,data,result,ax,legend):
        epi = result['xdict']['epi_plot']
        a = result['xdict']['a']
        for i,ens in enumerate(params_chipt['ensembles']):
            iens = params_chipt['ens_idx'][ens]
            ei = data['epi_b0'][iens]
            clr = params_plot['e_clr'][ens]
            alpha = 1
            mkr = params_plot['e_mrkr'][ens]
            dfv = gafit.dgaFV(ei,params_chipt['mpiL'][ens],result['ga_min'].values['g0fv'])
            gi = data['ga_b0'][iens]
            dgi = data['ga_bs'][:,iens].std()
            if type(epi) != np.ndarray and type(a) == np.ndarray:
                fv_shift = -.01
                xi = params_chipt['aw0'][ens]**2
                dxi = 2*params_chipt['aw0'][ens]*params_chipt['daw0'][ens]
                lbl = params_plot['m_lbl'][ens]
                l_check = params_plot['m_i']
                leg = ax.errorbar(-xi,gi-dfv,xerr=dxi,yerr=dgi,\
                    marker=mkr,color='k',mec='k',mfc='k',alpha=alpha,\
                    linestyle='None',label=lbl)
            elif type(a) != np.ndarray and type(epi) == np.ndarray:
                fv_shift = -.003
                xi = data['epi_b0'][iens]
                dxi = data['epi_bs'].std(axis=0)[iens]
                lbl = params_plot['a_lbl'][ens]
                l_check = params_plot['a_i']
                leg = ax.errorbar(-xi,gi-dfv,yerr=dgi,\
                    marker=mkr,color=clr,mec=clr,mfc=clr,alpha=alpha,\
                    linestyle='None',label=lbl)
            ax.errorbar(xi,gi-dfv,xerr=dxi,yerr=dgi,\
                marker=mkr,color=clr,mec=clr,mfc=clr,alpha=alpha,\
                linestyle='None')
            if args.show_fv:
                ax.errorbar(xi+fv_shift,gi,xerr=dxi,yerr=dgi,\
                    marker=mkr,color='k',mec='k',mfc='None',alpha=0.5,linestyle='None')
            if ens in l_check:
                legend.insert(3,leg)
        return legend
    def finish_plot(args,params_chipt,params_plot,result,ax,leg1,leg2):
        epi = result['xdict']['epi_plot']
        a = result['xdict']['a']
        if type(epi) != np.ndarray and type(a) == np.ndarray:
            ax.set_xlabel(r'$\epsilon_\pi = (a/w_0)^2$',fontsize=params_plot['fs'])
            ax.axis([args.asq_x[0],args.asq_x[1],args.asq_y[0],args.asq_y[1]])
            x0 = 0
        elif type(a) != np.ndarray and type(epi) == np.ndarray: 
            ax.set_xlabel(r'$\epsilon_\pi = m_\pi /(4\pi F_\pi)$',fontsize=params_plot['fs'])
            ax.vlines(params_chipt['epi_phys'],0.5,1.6,linestyle='--',color='k')
            ax.axis([args.epi_x[0],args.epi_x[1],args.epi_y[0],args.epi_y[1]])
            x0 = params_chipt['epi_phys']
        ax.set_ylabel(r'$g_A$',fontsize=params_plot['fs'])
        leg = ax.errorbar(x0,params_chipt['ga_phys'],\
            yerr=params_chipt['dga_phys'],\
            marker='o',markersize=10,mec='k',mfc='None',color='k',alpha=1,linestyle='None',\
            label=r'$g_A^{PDG}=%.4f(%s)$' \
                %(params_chipt['ga_phys'],str(params_chipt['dga_phys']).split('0')[-1]))
        leg2.append(leg)
        d_leg = ax.legend(handles=leg1,loc=4,numpoints=1,ncol=2,shadow=True,fancybox=True)
        plt.gca().add_artist(d_leg)
        ax.legend(handles=leg2,loc=3,numpoints=1,ncol=1,shadow=True,fancybox=True)
        ax.tick_params(axis='both', which='major', labelsize=14)
        return 0
    def fv_plot(args,params_chipt,params_plot,result,data,ax,select):
        e0 = result['xdict']['epi0']
        x = result['xdict']['xplot']
        # data for plots
        i_fv = [params_chipt['ens_idx']['a12m220S'],params_chipt['ens_idx']['a12m220'],\
            params_chipt['ens_idx']['a12m220L']]
        mL = [params_chipt['mpiL']['a12m220S'],params_chipt['mpiL']['a12m220'],\
            params_chipt['mpiL']['a12m220L']]
        epiL = [data['epi_b0'][params_chipt['ens_idx']['a12m220S']],\
            data['epi_b0'][params_chipt['ens_idx']['a12m220']],\
            data['epi_b0'][params_chipt['ens_idx']['a12m220L']]]
        xL = np.exp(-np.array(mL)) / np.sqrt(np.array(mL))
        epi = np.mean(epiL)
        epifv = epi
        if 'esq' in select:
            epi = epi**2
        a = params_chipt['aw0']['a12m220']
        # reconstructed fit
        mLplot = result['xdict']['mL']
        xplot = np.exp(-mLplot) / np.sqrt(mLplot)
        #print ga_L
        ga_L = np.zeros_like(mLplot)
        dga_L = np.zeros_like(mLplot)
        cov = np.array(result['ga_min'].matrix(correlation=False,skip_fixed=True))
        for i,mLi in enumerate(mLplot):
            ga_L[i] += gafit.dgaFV(epifv,mLi,result['ga_min'].values['g0fv'])
            dga_L[i] = gafit.dga_epi_fv(e0,epi,epifv,a,mLi,cov,**result['ga_min'].values)
        ga_L += gafit.ga_epi(epi0=e0,epi=epi,a=a,**result['ga_min'].values)
        #print ga_L
        gn = params_plot['e_clr']['a12m220']
        mkr = params_plot['e_mrkr']['a12m220']
        ax.fill_between(xplot,ga_L-dga_L, ga_L+dga_L,color=gn,alpha=0.2)
        ax.plot(xplot,ga_L,color='k',linestyle='--',label=r'NLO $\chi$PT prediction')
        for ii,i in enumerate(i_fv):
            gi = data['ga_b0'][i]
            dgi = data['ga_bs'][:,i].std()
            ax.errorbar(xL[ii],gi,yerr=dgi,color=gn,mec=gn,mfc=gn,marker=mkr)
        ax.set_ylabel(r'$g_A$',fontsize=params_plot['fs'])
        ax.set_xlabel(r'$e^{-m_\pi L} / \sqrt{m_\pi L}$',fontsize=params_plot['fs'])
        ax.axis([0.,0.024,1.2025,1.3125])
        ax.legend(loc=2,shadow=True,fancybox=True,fontsize=params_plot['fs'])
        
    ############################
    # END                      #
    ############################
    if args.fits in ['all','taylor_esq_1'] and args.plot:
        # select results
        select = 'taylor_esq_1'
        result = rdict[select].copy()
        ############################################
        # gA vs e_pi plot
        ############################################
        print('gA vs epi: Taylor e_pi^2')
        # initialize figure
        plt.figure('gA vs epi Taylor epsq',figsize=params_plot['fig_gldn'])
        ga_mpi_ax = plt.axes(params_plot['ga_axes'])
        leg1 = []
        leg2 = []
        # define x dependence
        result['xdict']['epi_plot'] = np.arange(0.001,0.41,.001)**2
        result['xdict']['xplot'] = np.arange(0.001,0.41,.001)
        result['xdict']['a'] = 0
        # continuum limit plot
        leg2 = continuum_plot(args,params_plot,result,ga_mpi_ax,leg2,select)
        # finite a plots
        leg1 = discrete_plot(args,params_plot,data,result,ga_mpi_ax,leg1,select)
        # add data points
        leg1 = data_plot(args,params_chipt,params_plot,data,result,ga_mpi_ax,leg1)
        # finish plot
        finish_plot(args,params_chipt,params_plot,result,ga_mpi_ax,leg1,leg2)
        ############################################
        # gA vs asq plot
        ############################################
        print('gA vs asq: Taylor e_pi^2')
        # initialize figure
        plt.figure('gA vs asq Taylor epsq',figsize=params_plot['fig_gldn'])
        ga_a_ax = plt.axes(params_plot['ga_axes'])
        leg1 = []
        leg2 = []
        result['xdict']['epi_plot'] = params_chipt['epi_phys']**2
        result['xdict']['xplot'] = np.arange(0,1.01,.01)**2
        result['xdict']['a'] = np.arange(0,1.01,.01)
        # continuum limit plot
        leg2 = continuum_plot(args,params_plot,result,ga_a_ax,leg2,select)
        # finite a plots
        leg1 = discrete_plot(args,params_plot,data,result,ga_a_ax,leg1,select)
        # add data points
        leg1 = data_plot(args,params_chipt,params_plot,data,result,ga_a_ax,leg1)
        # finish plot
        finish_plot(args,params_chipt,params_plot,result,ga_a_ax,leg1,leg2)
        ############################################
        # gA vs L plot
        ############################################
        print('gA vs L: Taylor e_pi^2')
        # initialize figure
        plt.figure('gA vs L Taylor epsq',figsize=params_plot['fig_gldn'])
        ga_L_ax = plt.axes(params_plot['mL_axes'])
        result['xdict']['epi0'] = args.e0**2
        result['xdict']['mL'] = np.arange(3,100.1,.1)
        fv_plot(args,params_chipt,params_plot,result,data,ga_L_ax,select)
        
    # display plot
    if args.plot:
        plt.ioff()
        if run_from_ipython():
            plt.show(block=False)
        else:
            plt.show()

if __name__=='__main__':
    # parse keyboard inputs
    args = parse_input()
    # set chipt parameters
    params_chipt = chipt_parameters()
    # set plotting parameters
    params_plot = plotting_parameters()
    # read data
    data = read_data(args,params_chipt)
    # fit data
    rdict = chipt_fit(args,params_chipt,data)
    # plot result
    plot = plot_fit(args,params_chipt,params_plot,data,rdict)


    
