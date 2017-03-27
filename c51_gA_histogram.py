# read bootstrapped data and construct histogram
import sqlite3
import json
import numpy as np
import ga_fit_funcs as ff
import data_params as dps
import matplotlib.pyplot as plt

def fit_list():
    # select nbs bootstraps from sqlite
    nbs = 5000
    # select model set here
    model_set = ['c0_nofv','t_esq0_a0','t_esq1_a0','t_esq0_a2','t_esq1_a2','x_nlo_a0','x_nlo_a2']
    #model_set = ['t_esq1_a0','t_esq0_a2','t_esq1_a2','x_nlo_a0','x_nlo_a2']
    title = dict()
    title['c0_nofv']      = r'Constant'
    title['t_esq0_a0']    = r'Taylor $C_0$ + FV'
    title['t_esq1_a2']    = r'Taylor $C_0+C_1\epsilon_\pi^2+a^2$'
    title['t_esq1_a0']    = r'Taylor $C_0+C_1\epsilon_\pi^2$'
    title['t_esq0_a2']    = r'Taylor $C_0+a^2$'
    title['x_nlo_a0']     = r'SU(2) NLO $\chi$PT w/o $a^2$'
    title['x_nlo_a2']     = r'SU(2) NLO $\chi$PT $+a^2$'
    return model_set, title, nbs
    

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False
plt.ion()

def read_sql(tblname,fitname,nbs):
    conn = sqlite3.connect('c51_ga.sqlite')
    c = conn.cursor()
    c.execute("SELECT nbs, result FROM %s WHERE name='%s' AND nbs<=%s;" %(tblname,fitname,nbs))
    mle = c.fetchall()
    mle_clean = dict()
    for i in range(len(mle)):
        mle_clean[mle[i][0]] = json.loads(mle[i][1])
    return mle_clean

def make_ga(mle,fitname):
    p = dps.gA_parameters()
    if fitname in ['c0_nofv','t_esq0_a0','t_esq1_a2','t_esq1_a0','t_esq0_a2']:
        gA = np.array([ff.ga_epi(epi0=mle[0]['e0']**2,epi=p['epi_phys']**2,a=0,**mle[i]) for i in range(len(mle))])
    elif fitname in ['x_nlo_a0','x_nlo_a2']:
        gA = np.array([ff.ga_su2(epi=p['epi_phys'],a=0,**mle[i]) for i in range(len(mle))])
    boot0 = gA[0]
    bootn = np.sort(gA[1:])
    return boot0, bootn

def CI68(bins, CI):
    bin68 = []
    for b in bins:
        if b >= CI[0] and b<= CI[1]:
            bin68.append(b)
        else:
            pass
    return bin68

def make_histogram(bssort, title, tag, weights=None, param=None, boot0=None, mk_hist=True, bootn_dict=False, w_dict=False):
    # get confidence interval index
    if type(weights) == np.ndarray:
        norm = sum(weights)
        CIidx = {0.025:0, 0.159:0, 0.250:0, 0.500:0, 0.750:0, 0.841:0, 0.975:0}
        s = 0
        i = 0
        while s <= 0.975*norm:
            s += weights[i]
            CIidx[0.975] = i
            if s <= 0.841344746*norm:
                CIidx[0.841] = i
                if s <= 0.75*norm:
                    CIidx[0.750] = i
                    if s <= 0.5*norm:
                        CIidx[0.500] = i
                        if s <= 0.25*norm:
                            CIidx[0.250] = i
                            if s <= 0.158655254*norm:
                                CIidx[0.159] = i
                                if s <= 0.025*norm:
                                    CIidx[0.025] = i
            i += 1
    else:
        norm = len(bssort)
        CIidx = {0.025:int(norm*0.025), 0.159:int(norm*0.158655254), 0.250:int(norm*0.25), 0.500:int(norm*0.5), 0.750:int(norm*0.75), 0.841:int(norm*0.841344746), 0.975:int(norm*0.975)}
    CI = [bssort[CIidx[0.159]], bssort[CIidx[0.841]], bssort[CIidx[0.500]]]

    if mk_hist:
        print('==================================================')
        print(title)
        print("nbs: %s" %len(bssort))
        print("median: %s" %CI[2])
        print("dCI: %s" %(0.5*(CI[1]-CI[0])))
        print("std: %s" %np.std(bssort))

        CI2s = [bssort[CIidx[0.025]], bssort[CIidx[0.975]], bssort[CIidx[0.500]]]

        # set histogram color
        color = '#b36ae2'
        # set binsize
        IQR = bssort[CIidx[0.750]] - bssort[CIidx[0.250]]
        binsize = 2.*IQR/(len(bssort)**(1./3.)) # Freedman-Diaconis rule
        setbins = int((bssort[-1]-bssort[0])/binsize)
        # start plot
        fig = plt.figure(figsize=(7,4.326237))
        ax = plt.axes([0.15,0.15,0.8,0.8])
        n, bins, patches = ax.hist(bssort, setbins, facecolor=color,ec='black',alpha=0.2,histtype='stepfilled',weights=weights)
        bin95 = CI68(bins, CI2s)
        n, bins, patches = ax.hist(bssort, bin95, facecolor=color,ec='black',alpha=0.5,histtype='stepfilled',weights=weights)
        bin68 = CI68(bins, CI)
        n, bins, patches = ax.hist(bssort, bin68, facecolor=color,ec='black',histtype='stepfilled',weights=weights)
        n, bins, patches = ax.hist(bssort, setbins, histtype='step',ec='black',weights=weights)
        n, bins, patches = ax.hist(bssort, bin95, histtype='step',ec='black',weights=weights)
        n, bins, patches = ax.hist(bssort, bin68, histtype='step',ec='black',weights=weights)
        if type(bootn_dict) == dict:
            for k in bootn_dict.keys():
                sortn = np.sort(bootn_dict[k])
                setbins = int((sortn[-1]-sortn[0])/binsize)
                ax.hist(bootn_dict[k], setbins, histtype='step',weights=np.ones_like(bootn_dict[k])*w_dict[k])
        x = np.delete(bins, -1)
        if param==None:
            ax.set_xlabel('$g_{A}$', fontsize=20)
        else:
            ax.set_xlabel('%s' %param, fontsize=20)
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=0)
        ax.set_title(title,x=0.15,y=0.68/0.8,fontsize=20,bbox=dict(facecolor=color))
        frame = plt.gca()
        frame.axes.get_yaxis().set_visible(False)
        plt.draw()
        plt.show()
        fig.savefig('gA_%s.pdf' %tag, format='pdf')
        return 0
    else:
        return np.mean(bssort),np.std(bssort)
        print("                 %s +- %s" %(CI[2],np.std(bssort)))


def akaike_weights(data0):
    AIC = []
    for k in data0.keys():
        AIC.append(data0[k]['AIC'])
    AIC = np.array(AIC)
    AICm = min(AIC)
    num = sum(np.exp(-0.5*(AIC-AICm)))
    w = dict()
    for k in data0.keys():
        w[k] = np.exp(-0.5*(data0[k]['AIC']-AICm))/num
        print('    %s;\taic %.3f:\tweight %.4f' %(k,data0[k]['AIC'],w[k]))
    return w

def model_avg(boot0,bootn,weights):
    cv = 0
    gAbs = np.zeros([len(boot0.keys()),len(bootn[bootn.keys()[0]])])
    for i,k in enumerate(boot0.keys()):
        cv += weights[k]*boot0[k]
        gAbs[i] = weights[k]*bootn[k]
    gAbs_w = gAbs.sum(axis=0)
    sdev = 0
    for k in bootn.keys():
        sdev += weights[k]*np.sqrt(np.var(bootn[k]) + (boot0[k] - cv)**2)
    aic_min = max(weights, key=lambda k:float(weights[k]))
    mle = read_sql('xcont',aic_min,nbs)
    boot0, bootn = make_ga(mle,aic_min)
    bs_avg,dgA_stat = make_histogram(bootn,title=title[aic_min],tag=aic_min,mk_hist=False)
    print('min %s:\t %s +- %s' %(title[aic_min],boot0,dgA_stat))
    print('AIC average result:      \t %s +- %s' %(cv, sdev))
    print('AIC average result:      \t %.4f +- %.4f +- %.4f'\
         %(cv,gAbs_w.std(),np.sqrt(sdev**2-gAbs_w.var())))
    return cv, sdev

def remake_ga(tag,title,nbs):
    mle = read_sql('xcont',tag,nbs)
    boot0, bootn = make_ga(mle,tag)
    make_histogram(bootn,title=title,tag=tag)
    return mle, boot0, bootn

def AICavg(model_set,mle,boot0,bootn):
    print('==================================================')
    print('AIC AVG gA')
    # construct akaike weights
    mle0 = dict()
    for m in model_set:
        mle0[m] = mle[m][0]
    w = akaike_weights(mle0)
    # avg models
    mean, sdev = model_avg(boot0,bootn,w)
    # make histogram
    whist = []
    cbootn = []
    for m in model_set:
        whist.append(w[m]*np.ones_like(bootn[m]))
        cbootn.append(bootn[m])
    whist = np.array(whist).flatten()
    cbootn = np.array(cbootn).flatten()
    idx = np.argsort(cbootn) # sort
    make_histogram(cbootn[idx],title='Akaike average',tag='AIC',weights=whist[idx],bootn_dict=bootn,w_dict=w)

if __name__=='__main__':
    model_set, title, nbs = fit_list()
    # remake gA for all selected models
    mle = dict()
    boot0 = dict()
    bootn = dict()
    for x in model_set:
        mle[x],boot0[x],bootn[x] = remake_ga(x,title[x],nbs)
    # calculate Akaike average
    AICavg(model_set,mle,boot0,bootn)

    plt.ioff()
    if run_from_ipython():
        plt.show(block=False)
    else:
        plt.show()
