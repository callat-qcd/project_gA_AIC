"""
This file performs the extrapolation analysis for the gA results using MDWF on HISQ ensembles
Required Python libraries/software
    numpy
    scipy
    matplotlib
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
    parser.add_argument('--error_x',default=True,action='store_false',\
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
    return p

def plotting_parameters():
    p = dict()
    # Figure sizes and fonts
    p['fig_gldn'] = (8.125,5.018)
    p['ga_axes'] = [0.095,0.128,0.895,0.865]
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
        return None
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
        y = self.ga_bs.mean(axis=0)
        x = (self.epi_bs**2).mean(axis=0)
        cdict = {'c0':c0, 'ca2':ca2, 'cm1':cm1}
        f = gafit.ga_epi(self.x0,x,self.p['xa'],**cdict)
        for i,ens in enumerate(self.p['ensembles']):
            f[i] += gafit.dgaFV(self.epi_b0[i],self.p['mL'][i],g0fv)
        if self.args.error_x:
            cov = np.zeros([self.p['Nbs'],self.p['l_d']])
            for i,ens in enumerate(self.p['ensembles']):
                dy = self.ga_bs[:,i] - y[i]
                df = gafit.ga_epi(self.x0,(self.epi_bs**2)[:,i],self.p['xa'][i],c0,ca2=ca2,cm1=cm1) - f[i]
                df += gafit.dgaFV(self.epi_bs[:,i],self.p['mL'][i],g0fv)
                cov[:,i]  = ( dy - df )**2
            cov = (1./self.p['Nbs']) * np.sum(cov,axis=0)
        else:
            cov = self.ga_bs.var(axis=0)
        '''
        y,f,cov are all len(l_d) arrays
        so numpy will properly do the multiplication/division
        '''
        chisq += np.sum( (y-f)**2 / cov )
        if self.args.g0fv != None:
            chisq += (g0fv - self.args.g0fv[0])**2 / self.args.g0fv[1]**2
        return chisq

def chipt_fit(args,p,data):
    def print_output(CS,ga_min,select):
        dof = CS.p['l_d'] - len(ga_min.values)
        print "chi^2 = %.4f, dof = %d, Q = %.4f" %(ga_min.fval,dof,gafit.Q(ga_min.fval,dof))
        for p in ga_min.parameters:
            print '  %s = %.4f +- %.4f' %(p,ga_min.values[p],ga_min.errors[p])

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
            dga_fit = gafit.dga_epi(epi0=x0,epi=xphys,a=0,lam_cov=cov,**ga_min.values)
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
    def continuum_plot(args,params_plot,result,ax,legend):
        e0 = result['xdict']['epi0']
        epi = result['xdict']['epi_plot']
        x = result['xdict']['xplot']
        # taylor fit needs g0fv, which infinite volume function doesn't know about
        # so chop of the last element (g0fv) of corrleation matrix
        cov = np.array(result['ga_min'].matrix(correlation=False,skip_fixed=True))[0:-1,0:-1]
        ga_plot = gafit.ga_epi(e0,epi,0,**result['ga_min'].values)
        dga_plot = gafit.dga_epi(epi0=e0,epi=epi,a=0,lam_cov=cov,**result['ga_min'].values)
        ax.fill_between(x,ga_plot-dga_plot,ga_plot+dga_plot,\
            color=params_plot['cont_color'],alpha=params_plot['a_cont'])
        leg, = ax.fill(x,-100*np.ones_like(result['xdict']['epi_plot']),\
            color=params_plot['cont_color'],alpha=params_plot['a_cont'],\
            label=r'$g_A^{LQCD}(\epsilon_\pi,a=0)$')
        legend.append(leg)
        return legend
    def discrete_plot(args,params_plot,result,ax,legend):
        e0 = result['xdict']['epi0']
        epi = result['xdict']['epi_plot']
        x = result['xdict']['xplot']
        cov = np.array(result['ga_min'].matrix(correlation=False,skip_fixed=True))[0:-1,0:-1]
        ga_plot15 = gafit.ga_epi(e0,epi,params_chipt['aw0']['a15m310'],**result['ga_min'].values)
        ga_plot12 = gafit.ga_epi(e0,epi,params_chipt['aw0']['a12m310'],**result['ga_min'].values)
        ga_plot09 = gafit.ga_epi(e0,epi,params_chipt['aw0']['a09m310'],**result['ga_min'].values)
        ga_plot_a = [ga_plot15,ga_plot12,ga_plot09]
        ga_plot_lbl = [r'$g_A(\epsilon_\pi,a=0.15)$',r'$g_A(\epsilon_\pi,a=0.12)$',
            r'$g_A(\epsilon_\pi,a=0.09)$']
        leg, = ax.plot(x,ga_plot15,color=params_plot['e_clr']['a15m310'],alpha=0.5,\
            label=r'$g_A(\epsilon_\pi,a=0.15)$')
        legend.insert(0,leg)
        leg, = ax.plot(x,ga_plot12,color=params_plot['e_clr']['a12m310'],alpha=0.5,\
            label=r'$g_A(\epsilon_\pi,a=0.12)$')
        legend.insert(0,leg)
        leg, = ax.plot(x,ga_plot09,color=params_plot['e_clr']['a09m310'],alpha=0.5,\
            label=r'$g_A(\epsilon_\pi,a=0.09)$')
        legend.insert(0,leg)
        return legend
    def data_plot(args,params_chipt,params_plot,data,result,ax,legend):
        fv_shift = -.003
        for i,ens in enumerate(params_chipt['ensembles']):
            lbl = params_plot['a_lbl'][ens]
            clr = params_plot['e_clr'][ens]
            alpha = 1
            mkr = params_plot['e_mrkr'][ens]
            ei = data['epi_b0'][i]
            dei = data['epi_bs'].std(axis=0)[i]
            dfv = gafit.dgaFV(ei,params_chipt['mpiL'][ens],result['ga_min'].values['g0fv'])
            gi = data['ga_b0'][i]
            dgi = data['ga_bs'][:,i].std()
            leg = ax.errorbar(ei,gi-dfv,xerr=dei,yerr=dgi,\
                marker=mkr,color=clr,mec=clr,mfc=clr,alpha=alpha,\
                linestyle='None',label=lbl)
            if args.show_fv:
                ax.errorbar(ei+fv_shift,gi,xerr=dei,yerr=dgi,\
                    marker=mkr,color='k',mec='k',mfc='None',alpha=0.5,linestyle='None')
            if ens in params_plot['a_i']:
                legend.insert(3,leg)
        return legend
    def finish_plot(args,params_chipt,params_plot,ax,leg1,leg2):
        ax.set_xlabel(r'$\epsilon_\pi = m_\pi /(4\pi F_\pi)$',fontsize=params_plot['fs'])
        ax.set_ylabel(r'$g_A$',fontsize=params_plot['fs'])
        leg = ax.errorbar(params_chipt['epi_phys'],params_chipt['ga_phys'],\
            yerr=params_chipt['dga_phys'],\
            marker='o',markersize=10,mec='k',mfc='None',color='k',alpha=1,linestyle='None',\
            label=r'$g_A^{PDG}=%.4f(%s)$' \
                %(params_chipt['ga_phys'],str(params_chipt['dga_phys']).split('0')[-1]))
        leg2.append(leg)
        ax.vlines(params_chipt['epi_phys'],0.5,1.6,linestyle='--',color='k')
        ax.axis([args.epi_x[0],args.epi_x[1],args.epi_y[0],args.epi_y[1]])
        d_leg = ga_mpi_ax.legend(handles=leg1,loc=4,numpoints=1,ncol=2,shadow=True,fancybox=True)
        plt.gca().add_artist(d_leg)
        ax.legend(handles=leg2,loc=3,numpoints=1,ncol=1,shadow=True,fancybox=True)
        ax.tick_params(axis='both', which='major', labelsize=14)
        return 0
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
        # continuum limit plot
        leg2 = continuum_plot(args,params_plot,result,ga_mpi_ax,leg2)
        # finite a plots
        leg1 = discrete_plot(args,params_plot,result,ga_mpi_ax,leg1)
        # add data points
        leg1 = data_plot(args,params_chipt,params_plot,data,result,ga_mpi_ax,leg1)
        # finish plot
        finish_plot(args,params_chipt,params_plot,ga_mpi_ax,leg1,leg2)
        ############################################
        # gA vs asq plot
        ############################################
        print('gA vs asq: Taylor e_pi^2')
        # initialize figure
        #plt.figure('gA vs asq Taylor epsq',figsize=params_plot['fig_gldn'])
        #ga_mpi_ax = plt.axes(params_plot['ga_axes'])
        leg1 = []
        leg2 = []
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

'''
############################################
# gA vs asq
############################################
fig += 1
print('gA vs asq: Taylor e_pi^2, Order 1, Fig=%d' %fig)
plt.figure(fig,figsize=fig_gldn)
ga_asq_ax = plt.axes(ga_axes)
leg1 = []
leg2 = []
aplot = np.arange(0,1.01,.01)
xplot = aplot**2
# physical pion mass plot
ga_asq = fit.ga_epi(x0,epi_phys**2,aplot,c0,ca2=ca2,cm1=cm1,cm2=cm2)
dga_asq = fit.dga_a(x0,epi_phys**2,aplot,c0,cov_lam[0:-1,0:-1],ca2=ca2,cm1=cm1,cm2=cm2)
ga_asq_ax.fill_between(xplot,ga_asq-dga_asq,ga_asq+dga_asq,\
    color=cont_color,alpha=a_cont)
leg, = ga_asq_ax.fill(xplot,-100*np.ones_like(aplot),\
    color=cont_color,alpha=a_cont,\
    label=r'$g_A^{LQCD}(\epsilon_\pi^\textrm{phys},a/w_0)$')
leg2.append(leg)
# unphysical mpi plots
esq310 = (epi_b0[0:3]**2).mean()
esq220 = (epi_b0[3:7]**2).mean()
esq130 = (epi_b0[7:]**2).mean()
ga_plot310 = fit.ga_epi(x0,esq310,aplot,c0,ca2=ca2,cm1=cm1,cm2=cm2)
ga_plot220 = fit.ga_epi(x0,esq220,aplot,c0,ca2=ca2,cm1=cm1,cm2=cm2)
ga_plot130 = fit.ga_epi(x0,esq130,aplot,c0,ca2=ca2,cm1=cm1,cm2=cm2)
ga_plot_m = [ga_plot310,ga_plot220,ga_plot130]
ga_plot_lbl = [r'$g_A(\epsilon_\pi^{(310)},a/w_0)$',r'$g_A(\epsilon_\pi^{(220)},a/w_0)$',
    r'$g_A(\epsilon_\pi^{(130)},a/w_0)$',r'$g_A(\epsilon_\pi,a=0.06)$']
leg, = ga_asq_ax.plot(xplot,ga_plot310,color='k',linestyle='-.',alpha=0.5,
    label=r'$g_A(\epsilon_\pi^{(310)},a/w_0)$')
leg1.insert(0,leg)
leg, = ga_asq_ax.plot(xplot,ga_plot220,color='k',linestyle='--',alpha=0.5,
    label=r'$g_A(\epsilon_\pi^{(220)},a/w_0)$')
leg1.insert(0,leg)
leg, = ga_asq_ax.plot(xplot,ga_plot130,color='k',linestyle='-',alpha=0.5,
    label=r'$g_A(\epsilon_\pi^{(130)},a/w_0)$')
leg1.insert(0,leg)


# add data points
fv_shift = -.01
for i,ens in enumerate(ensembles):
    lbl = m_lbl[ens]
    clr = e_clr[ens]
    alpha = 1
    mkr = e_mrkr[ens]
    dfv = fit.dgaFV(epi_b0[i],mL[i],g0fv)
    ai = aw0[ens]**2
    dai = 2*aw0[ens]*daw0[ens]
    gi = ga_b0[i]
    dgi = ga_bs.std(axis=0)[i]
    ga_asq_ax.errorbar(ai,gi-dfv,yerr=dgi,\
        marker=mkr,color=clr,mec=clr,mfc=clr,alpha=alpha,\
        linestyle='None',label=lbl)
    leg = ga_asq_ax.errorbar(-ai,gi-dfv,yerr=dgi,\
        marker=mkr,color='k',mec='k',mfc='k',alpha=alpha,\
        linestyle='None',label=lbl)
    if args.show_fv:
        ga_asq_ax.errorbar(ai+fv_shift,gi,xerr=dai,yerr=dgi,\
            marker=mkr,color='k',mec='k',mfc='None',alpha=0.5,linestyle='None')
    if ens in m_i:
        leg1.insert(3,leg)
ga_asq_ax.set_xlabel(r'$(a/w_0)^2$',fontsize=fs)
ga_asq_ax.set_ylabel(r'$g_A$',fontsize=fs)
leg = ga_asq_ax.errorbar(0,ga_phys,yerr=dga_phys,\
    marker='o',markersize=10,mec='k',mfc='None',color='k',alpha=1,linestyle='None',\
    label=r'$g_A^{PDG}=%.4f(%s)$' %(ga_phys,str(dga_phys).split('0')[-1]))
leg2.append(leg)

ga_asq_ax.axis([args.asq_x[0],args.asq_x[1],args.asq_y[0],args.asq_y[1]])

d_leg = ga_asq_ax.legend(handles=leg1,loc=4,numpoints=1,ncol=2,shadow=True,fancybox=True)
plt.gca().add_artist(d_leg)

ga_asq_ax.legend(handles=leg2,loc=3,numpoints=1,ncol=1,shadow=True,fancybox=True)
ga_asq_ax.tick_params(axis='both', which='major', labelsize=14)
'''
    
