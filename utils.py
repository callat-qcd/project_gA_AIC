import matplotlib.pyplot as plt
import tables as h5
import numpy as np
import ga_fit_funcs as gafit

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False
def read_data(fname,args,p):
    data = dict()
    c51_data = h5.open_file(fname)
    # check how many Nbs are in file
    if args.Nbs == None:
        Nbs = c51_data.get_node('/gA/'+p['ensembles'][0]+'/bs').read().shape[0]
    else:
        Nbs = args.Nbs
    p['Nbs'] = Nbs
    print('using Nbs = %d samples' %Nbs)

    ga_bs    = np.zeros([Nbs,p['l_d']])
    ga_b0    = np.zeros([p['l_d']])
    epi_bs   = np.zeros_like(ga_bs)
    epi_b0   = np.zeros_like(ga_b0)
    mL_b0    = np.zeros([p['l_d']])
    mL_bs    = np.zeros([Nbs,p['l_d']])
    aw0_b0   = np.zeros([p['l_d']])
    aw0_bs   = np.zeros([Nbs,p['l_d']])
    aSaw0_b0 = np.zeros([p['l_d']])
    aSaw0_bs = np.zeros([Nbs,p['l_d']])
    eju_bs   = np.zeros_like(ga_bs)
    eju_b0   = np.zeros_like(ga_b0)
    epqsq_bs = np.zeros_like(ga_bs)
    epqsq_b0 = np.zeros_like(ga_b0)
    for i,ens in enumerate(p['ensembles']):
        ga_bs[:,i]    = c51_data.get_node('/gA/'+ens+'/bs').read()[0:Nbs]
        ga_b0[i]      = float(c51_data.get_node('/gA/'+ens+'/b0').read())
        epi_bs[:,i]   = c51_data.get_node('/epi/'+ens+'/bs').read()[0:Nbs]
        epi_b0[i]     = float(c51_data.get_node('/epi/'+ens+'/b0').read())
        mL_bs[:,i]    = c51_data.get_node('/mpiL/'+ens+'/bs').read()[0:Nbs]
        mL_b0[i]      = float(c51_data.get_node('/mpiL/'+ens+'/b0').read())
        aw0_bs[:,i]   = c51_data.get_node('/aw0/'+ens+'/bs').read()[0:Nbs]
        aw0_b0[i]     = float(c51_data.get_node('/aw0/'+ens+'/b0').read())
        # we have to multiply a by sqrt(alpha_S) as a is squared in the extrapolation functions
        # to swap sqrt(alpha_S) a in for a 
        aSaw0_bs[:,i] = aw0_bs[:,i] * np.sqrt(p['afs'][ens])
        aSaw0_b0[i]   = aw0_b0[i] * np.sqrt(p['afs'][ens])
        eju_bs[:,i]   = c51_data.get_node('/eju/'+ens+'/bs').read()[0:Nbs]
        eju_b0[i]     = float(c51_data.get_node('/eju/'+ens+'/b0').read())
        epqsq_bs[:,i] = c51_data.get_node('/epqsq/'+ens+'/bs').read()[0:Nbs]
        epqsq_b0[i]   = float(c51_data.get_node('/epqsq/'+ens+'/b0').read())
        print(ens,ga_b0[i],ga_bs.std(axis=0)[i],epi_b0[i],epi_bs.std(axis=0)[i])
    data['ga_bs']    = ga_bs
    data['ga_b0']    = ga_b0
    data['epi_bs']   = epi_bs
    data['epi_b0']   = epi_b0
    data['mL_bs']    = mL_bs
    data['mL_b0']    = mL_b0
    data['aw0_bs']   = aw0_bs
    data['aw0_b0']   = aw0_b0
    data['eju_b0']   = eju_b0
    data['eju_bs']   = eju_bs
    data['epqsq_b0'] = epqsq_b0
    data['epqsq_bs'] = epqsq_bs
    data['aSaw0_bs'] = aSaw0_bs
    data['aSaw0_b0'] = aSaw0_b0
    c51_data.close()
    return data

def plot_fit(args,params_chipt,params_plot,data,rdict):
    ############################
    # FUNCTIONS FOR plot_fit() #
    ############################
    def continuum_plot(args,params_plot,result,ax,legend,select):
        epi = result['xdict']['epi_plot']
        x = result['xdict']['xplot']
        a = result['xdict']['a']
        if type(epi) != np.ndarray and type(a) == np.ndarray:
            label = r'$g_A^{LQCD}(\epsilon_\pi^{phys},a/w_0)$'
        elif type(a) != np.ndarray and type(epi) == np.ndarray:
            label = r'$g_A^{LQCD}(\epsilon_\pi,a=0)$'
        # taylor fit needs g0fv, which infinite volume function doesn't know about
        # so chop of the last element (g0fv) of corrleation matrix
        cov = np.array(result['ga_min'].matrix(correlation=False,skip_fixed=True))
        if select in ['t_esq0_a0','t_esq1_a2','t_esq0_a2','t_esq1_a0','t_esq1_a2_ea2']:
            e0 = result['xdict']['epi0']
            ga_plot = gafit.ga_epi(epi0=e0,epi=epi,a=a,**result['ga_min'].values)
            cov2 = cov[0:-1,0:-1]
            dga_plot = gafit.dga_epi(epi0=e0,epi=epi,a=a,lam_cov=cov2,**result['ga_min'].values)
        elif select in ['x_nlo_a0','x_nlo_a2','x_nlo_aSa2','x_nlo_a2_ea2']:
            ga_plot = gafit.ga_su2(epi=epi,a=a,**result['ga_min'].values)
            dga_plot = gafit.dga_su2(epi=epi,a=a,lam_cov=cov,**result['ga_min'].values)
        elif select in ['xma_nlo_a2']:
            cov2 = cov[0:-1,0:-1]
            ga_plot = gafit.ga_su2(epi=epi,a=a,**result['ga_min'].values)
            dga_plot = gafit.dga_su2(epi=epi,a=a,lam_cov=cov2,**result['ga_min'].values)
        leg, = ax.fill(x,-100*np.ones_like(x),\
            color=params_plot['cont_color'],alpha=params_plot['a_cont'],label=label)
        ax.fill_between(x,ga_plot-dga_plot,ga_plot+dga_plot,\
            color=params_plot['cont_color'],alpha=params_plot['a_cont'])
        legend.append(leg)
        return legend
    def discrete_plot(args,params_plot,data,result,ax,legend,select):
        epi = result['xdict']['epi_plot']
        x = result['xdict']['xplot']
        a = result['xdict']['a']
        if type(epi) != np.ndarray and type(a) == np.ndarray:
            e = np.array([data['epi_b0'][params_chipt['ens_idx']['a15m310']],\
                data['epi_b0'][params_chipt['ens_idx']['a15m220']],\
                data['epi_b0'][params_chipt['ens_idx']['a15m130']]])
            ga_plot_lbl = [r'$g_A(\epsilon_\pi^{(310)},a/w_0)$',
                r'$g_A(\epsilon_\pi^{(220)},a/w_0)$',r'$g_A(\epsilon_\pi^{(130)},a/w_0)$']
            color = ['k','k','k']
            ls = ['-.','--','-']
        elif type(a) != np.ndarray and type(epi) == np.ndarray:
            ga_plot_lbl = [r'$g_A(\epsilon_\pi,a=0.15)$',
                r'$g_A(\epsilon_\pi,a=0.12)$',r'$g_A(\epsilon_\pi,a=0.09)$']
            color = [params_plot['e_clr']['a15m310'],
                params_plot['e_clr']['a12m310'],params_plot['e_clr']['a09m310']]
            ls = ['-','-','-']
        if select in ['t_esq0_a2','t_esq1_a2','t_esq1_a0','t_esq1_a2_ea2']:
            e0 = result['xdict']['epi0']
            if type(epi) != np.ndarray and type(a) == np.ndarray:
                if 'esq' in select or 't_esq0_a2' in select:
                    ep = e**2
                else: ep = e
                ga_0 = gafit.ga_epi(epi0=e0,epi=ep[0],a=a,**result['ga_min'].values)
                ga_1 = gafit.ga_epi(epi0=e0,epi=ep[1],a=a,**result['ga_min'].values)
                ga_2 = gafit.ga_epi(epi0=e0,epi=ep[2],a=a,**result['ga_min'].values)
            elif type(a) != np.ndarray and type(epi) == np.ndarray:
                a0 = params_chipt['aw0']['a15m310']
                a1 = params_chipt['aw0']['a12m310']
                a2 = params_chipt['aw0']['a09m310']
                ga_0 = gafit.ga_epi(epi0=e0,epi=epi,a=a0,**result['ga_min'].values)
                ga_1 = gafit.ga_epi(epi0=e0,epi=epi,a=a1,**result['ga_min'].values)
                ga_2 = gafit.ga_epi(epi0=e0,epi=epi,a=a2,**result['ga_min'].values)
        elif select in ['x_nlo_a2','x_nlo_aSa2','x_nlo_a2_ea2']:
            if type(epi) != np.ndarray and type(a) == np.ndarray:
                ga_0 = gafit.ga_su2(epi=e[0],a=a,**result['ga_min'].values)
                ga_1 = gafit.ga_su2(epi=e[1],a=a,**result['ga_min'].values)
                ga_2 = gafit.ga_su2(epi=e[2],a=a,**result['ga_min'].values)
            elif type(a) != np.ndarray and type(epi) == np.ndarray:
                ga_0 = gafit.ga_su2(epi,params_chipt['aw0']['a15m310'],**result['ga_min'].values)
                ga_1 = gafit.ga_su2(epi,params_chipt['aw0']['a12m310'],**result['ga_min'].values)
                ga_2 = gafit.ga_su2(epi,params_chipt['aw0']['a09m310'],**result['ga_min'].values)
        ga_plot_a = [ga_0,ga_1,ga_2]
        leg, = ax.plot(x,ga_0,color=color[0],alpha=0.5,label=ga_plot_lbl[0],ls=ls[0])
        legend.insert(0,leg)
        leg, = ax.plot(x,ga_1,color=color[1],alpha=0.5,label=ga_plot_lbl[1],ls=ls[1])
        legend.insert(0,leg)
        leg, = ax.plot(x,ga_2,color=color[2],alpha=0.5,label=ga_plot_lbl[2],ls=ls[2])
        legend.insert(0,leg)
        return legend
    def data_plot(args,params_chipt,params_plot,data,result,ax,legend):
        epi = result['xdict']['epi_plot']
        a = result['xdict']['a']
        if 'g0fv' in result['ga_min'].values:
            g0fv = result['ga_min'].values['g0fv']
        else:
            g0fv = result['ga_min'].values['g0']
        for i,ens in enumerate(params_chipt['ensembles']):
            iens = params_chipt['ens_idx'][ens]
            ei = data['epi_b0'][iens]
            clr = params_plot['e_clr'][ens]
            alpha = 1
            mkr = params_plot['e_mrkr'][ens]
            fv_class = gafit.FV_function(ei,params_chipt['mpiL'][ens])
            dfv = fv_class.dgaFV(g0fv)
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
            ax.set_xlabel(r'$(a/w_0)^2$',fontsize=params_plot['fs'])
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
        d_leg = ax.legend(handles=leg1,loc=4,numpoints=1,ncol=2,shadow=False,fancybox=True)
        plt.gca().add_artist(d_leg)
        ax.legend(handles=leg2,loc=1,numpoints=1,ncol=1,shadow=False,fancybox=True)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title(plt.get_figlabels()[-1].split(' ')[-1],fontdict={'fontsize':20,'verticalalignment':'top','horizontalalignment':'left'},x=0.05,y=0.9)
        plt.savefig('%s.pdf' %(plt.get_figlabels()[-1].replace(' ','_').replace('(','').replace(')','').replace('$','').replace('\\','').replace('.','').replace('^','').replace('/','')), transparent=True)
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
        fv_class = gafit.FV_function(epifv,mLplot)
        if 'g0fv' in result['ga_min'].values:
            ga_L  = fv_class.dgaFV(result['ga_min'].values['g0fv'])
            ga_L += gafit.ga_epi(epi0=e0,epi=epi,a=a,**result['ga_min'].values)
        else:
            ga_L  = fv_class.dgaFV(result['ga_min'].values['g0'])        
            ga_L += gafit.ga_su2(epi=epi,a=a,**result['ga_min'].values)
        dga_L = np.zeros_like(ga_L)
        for i,mLi in enumerate(mLplot):
            if select in ['t_esq1_a2','t_esq0_a2','t_esq1_a2_ea2','t_esq1_a0']:
                dga_L[i] = gafit.dga_epi_fv(e0,epi,epifv,a,mLi,cov,**result['ga_min'].values)
            elif select in ['x_nlo_a0','x_nlo_a2','x_nlo_a2','x_nlo_aSa2','x_nlo_a2_ea2']:
                dga_L[i] = gafit.dfv_su2_nlo(epi,mLi,a,lam_cov=cov,**result['ga_min'].values)
        gn = params_plot['e_clr']['a12m220']
        mkr = params_plot['e_mrkr']['a12m220']
        ax.fill_between(xplot,ga_L-dga_L, ga_L+dga_L,color=gn,alpha=0.2)
        ax.plot(xplot,ga_L,color='k',linestyle='--',label=r'NLO $\chi$PT prediction')
        for ii,i in enumerate(i_fv):
            gi = data['ga_b0'][i]
            dgi = data['ga_bs'][:,i].std()
            ax.errorbar(xL[ii],gi,yerr=dgi,color=gn,mec=gn,mfc=gn,marker=mkr)
        ax.set_ylabel(r'$g_A$',fontsize=params_plot['fs'])
        ax.set_xlabel(r'$e^{-m_\pi L} / (m_\pi L)^{1/2}$',fontsize=params_plot['fs'])
        ax.axis([0.,0.024,1.2025,1.3125])
        ax.legend(loc=3,shadow=False,fancybox=True,fontsize=params_plot['fs'])
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title(plt.get_figlabels()[-1].split(' ')[-1],fontdict={'fontsize':20,'verticalalignment':'top','horizontalalignment':'left'},x=0.05,y=0.9)
        plt.savefig('%s.pdf' %(plt.get_figlabels()[-1].replace(' ','_').replace('(','').replace(')','').replace('$','').replace('\\','').replace('.','').replace('^','').replace('/','')), transparent=True)
        
    ############################
    # END                      #
    ############################
    if args.fits in ['all','t_esq1_a2'] and args.plot:
        # select results
        select = 't_esq1_a2'
        result = rdict[select].copy()
        print('plotting %s' %select)
        ############################################
        # gA vs e_pi plot
        ############################################
        #print('gA vs epi: Taylor e_pi^2')
        # initialize figure
        plt.figure('$g_A$ vs. $\epsilon_\pi$ T$\epsilon_\pi^2a^2$',figsize=params_plot['fig_gldn'])
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
        #print('gA vs asq: Taylor e_pi^2')
        # initialize figure
        plt.figure('$g_A$ vs. $(a/w_0)^2$ T$\epsilon_\pi^2a^2$',figsize=params_plot['fig_gldn'])
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
        #print('gA vs L:   Taylor e_pi^2')
        # initialize figure
        plt.figure('$g_A$ vs. $m_\pi L$ T$\epsilon_\pi^2a^2$',figsize=params_plot['fig_gldn'])
        ga_L_ax = plt.axes(params_plot['mL_axes'])
        result['xdict']['epi0'] = args.e0**2
        result['xdict']['mL'] = np.arange(3,100.1,.1)
        fv_plot(args,params_chipt,params_plot,result,data,ga_L_ax,select)
    if args.fits in ['all','x_nlo_a2'] and args.plot:
        # select results
        select = 'x_nlo_a2'
        result = rdict[select].copy()
        print('plotting %s' %select)
        ############################################
        # gA vs e_pi plot
        ############################################
        #print('gA vs epi: SU(2) NLO')
        # initialize figure
        plt.figure('$g_A$ vs. $\epsilon_\pi$ $\chi\epsilon_\pi^2a^2$',figsize=params_plot['fig_gldn'])
        ga_mpi_ax = plt.axes(params_plot['ga_axes'])
        leg1 = []
        leg2 = []
        # define x dependence
        result['xdict']['epi_plot'] = np.arange(0.001,0.41,.001)
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
        #print('gA vs asq: SU(2) NLO')
        # initialize figure
        plt.figure('$g_A$ vs. $(a/w_0)^2$ $\chi\epsilon_\pi^2a^2$',figsize=params_plot['fig_gldn'])
        ga_a_ax = plt.axes(params_plot['ga_axes'])
        leg1 = []
        leg2 = []
        result['xdict']['epi_plot'] = params_chipt['epi_phys']
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
        #print('gA vs L:   SU(2) NLO')
        # initialize figure
        plt.figure('$g_A$ vs. $m_\pi L$ $\chi\epsilon_\pi^2a^2$',figsize=params_plot['fig_gldn'])
        ga_L_ax = plt.axes(params_plot['mL_axes'])
        result['xdict']['epi0'] = args.e0**2
        result['xdict']['mL'] = np.arange(3,100.1,.1)
        fv_plot(args,params_chipt,params_plot,result,data,ga_L_ax,select)
    if args.fits in ['all','t_esq0_a2'] and args.plot:
        # select results
        select = 't_esq0_a2'
        result = rdict[select].copy()
        print('plotting %s' %select)
        ############################################
        # gA vs e_pi plot
        ############################################
        #print('gA vs epi: Taylor0 epsq')
        # initialize figure
        plt.figure('gA vs epi Taylor0 epsq',figsize=params_plot['fig_gldn'])
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
        #print('gA vs asq: Taylor0 e_pi^2')
        # initialize figure
        plt.figure('gA vs asq Taylor0 epsq',figsize=params_plot['fig_gldn'])
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
        #print('gA vs L:   Taylor0 e_pi^2')
        # initialize figure
        plt.figure('gA vs L Taylor0 epsq',figsize=params_plot['fig_gldn'])
        ga_L_ax = plt.axes(params_plot['mL_axes'])
        result['xdict']['epi0'] = args.e0**2
        result['xdict']['mL'] = np.arange(3,100.1,.1)
        fv_plot(args,params_chipt,params_plot,result,data,ga_L_ax,select)
    if args.fits in ['all','t_esq1_a0'] and args.plot:
        # select results
        select = 't_esq1_a0'
        result = rdict[select].copy()
        print('plotting %s' %select)
        ############################################
        # gA vs e_pi plot
        ############################################
        #print('gA vs epi: Taylor epsq a0')
        # initialize figure
        plt.figure('gA vs epi Taylor epsq a0',figsize=params_plot['fig_gldn'])
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
        #print('gA vs asq: Taylor epsq a0')
        # initialize figure
        plt.figure('gA vs asq Taylor epsq a0',figsize=params_plot['fig_gldn'])
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
        #print('gA vs L:   Taylor epsq a0')
        # initialize figure
        plt.figure('gA vs L Taylor epsq a0',figsize=params_plot['fig_gldn'])
        ga_L_ax = plt.axes(params_plot['mL_axes'])
        result['xdict']['epi0'] = args.e0**2
        result['xdict']['mL'] = np.arange(3,100.1,.1)
        fv_plot(args,params_chipt,params_plot,result,data,ga_L_ax,select)
    if args.fits in ['other','x_nlo_a2_ea2'] and args.plot:
        # select results
        select = 'x_nlo_a2_ea2'
        result = rdict[select].copy()
        print('plotting %s' %select)
        ############################################
        # gA vs e_pi plot
        ############################################
        #print('gA vs epi: SU(2) NLO + a**2 epi**2')
        # initialize figure
        plt.figure('gA vs epi SU2 NLO aesq',figsize=params_plot['fig_gldn'])
        ga_mpi_ax = plt.axes(params_plot['ga_axes'])
        leg1 = []
        leg2 = []
        # define x dependence
        result['xdict']['epi_plot'] = np.arange(0.001,0.41,.001)
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
        #print('gA vs asq: SU(2) NLO + a**2 epi**2')
        # initialize figure
        plt.figure('gA vs asq SU2 NLO aesq',figsize=params_plot['fig_gldn'])
        ga_a_ax = plt.axes(params_plot['ga_axes'])
        leg1 = []
        leg2 = []
        result['xdict']['epi_plot'] = params_chipt['epi_phys']
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
        #print('gA vs L:   SU(2) NLO')
        # initialize figure
        plt.figure('gA vs L SU2 NLO aesq',figsize=params_plot['fig_gldn'])
        ga_L_ax = plt.axes(params_plot['mL_axes'])
        result['xdict']['epi0'] = args.e0**2
        result['xdict']['mL'] = np.arange(3,100.1,.1)
        fv_plot(args,params_chipt,params_plot,result,data,ga_L_ax,select)
    if args.fits in ['other','t_esq1_a2_ea2'] and args.plot:
        # select results
        select = 't_esq1_a2_ea2'
        result = rdict[select].copy()
        print('plotting %s' %select)
        ############################################
        # gA vs e_pi plot
        ############################################
        #print('gA vs epi: Taylor e_pi^2 + asq epsq')
        # initialize figure
        plt.figure('gA vs epi Taylor epsq + asq epsq',figsize=params_plot['fig_gldn'])
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
        #print('gA vs asq: Taylor e_pi^2 + asq epsq')
        # initialize figure
        plt.figure('gA vs asq Taylor epsq + asq epsq',figsize=params_plot['fig_gldn'])
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
        #print('gA vs L:   Taylor e_pi^2 + asq epsq')
        # initialize figure
        plt.figure('gA vs L Taylor epsq + asq epsq',figsize=params_plot['fig_gldn'])
        ga_L_ax = plt.axes(params_plot['mL_axes'])
        result['xdict']['epi0'] = args.e0**2
        result['xdict']['mL'] = np.arange(3,100.1,.1)
        fv_plot(args,params_chipt,params_plot,result,data,ga_L_ax,select)
    if args.fits in ['other','xma_nlo_a2'] and args.plot:
        # select results
        select = 'xma_nlo_a2'
        result = rdict[select].copy()
        print('plotting %s' %select)
        ############################################
        # gA vs e_pi plot
        ############################################
        #print('gA vs epi: MA SU(2) NLO')
        # initialize figure
        plt.figure('gA vs epi MA SU(2) NLO',figsize=params_plot['fig_gldn'])
        ga_mpi_ax = plt.axes(params_plot['ga_axes'])
        leg1 = []
        leg2 = []
        # define x dependence
        result['xdict']['epi_plot'] = np.arange(0.001,0.41,.001)
        result['xdict']['xplot'] = np.arange(0.001,0.41,.001)
        result['xdict']['a'] = 0
        # continuum limit plot
        leg2 = continuum_plot(args,params_plot,result,ga_mpi_ax,leg2,select)
        # finite a plots
        #leg1 = discrete_plot(args,params_plot,data,result,ga_mpi_ax,leg1,select)
        # add data points
        #leg1 = data_plot(args,params_chipt,params_plot,data,result,ga_mpi_ax,leg1)
        # finish plot
        finish_plot(args,params_chipt,params_plot,result,ga_mpi_ax,leg1,leg2)
        '''
        ############################################
        # gA vs asq plot
        ############################################
        #print('gA vs asq: SU(2) NLO')
        # initialize figure
        plt.figure('gA vs asq SU(2) NLO',figsize=params_plot['fig_gldn'])
        ga_a_ax = plt.axes(params_plot['ga_axes'])
        leg1 = []
        leg2 = []
        result['xdict']['epi_plot'] = params_chipt['epi_phys']
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
        #print('gA vs L:   SU(2) NLO')
        # initialize figure
        plt.figure('gA vs L SU(2) NLO',figsize=params_plot['fig_gldn'])
        ga_L_ax = plt.axes(params_plot['mL_axes'])
        result['xdict']['epi0'] = args.e0**2
        result['xdict']['mL'] = np.arange(3,100.1,.1)
        fv_plot(args,params_chipt,params_plot,result,data,ga_L_ax,select)
        '''
    # display plot
    if args.plot:
        plt.ioff()
        if run_from_ipython():
            plt.show(block=False)
        else:
            plt.show()
