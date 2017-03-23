# read bootstrapped data and construct histogram
import sqlite3
import json
import numpy as np
import ga_fit_funcs as ff
import data_params as dps
import matplotlib.pyplot as plt

def read_sql(tblname,fitname):
    conn = sqlite3.connect('c51_ga.sqlite')
    #conn.enable_load_extension(True)
    #conn.load_extension("./json1")
    c = conn.cursor()
    c.execute("SELECT nbs, result FROM %s WHERE name='%s';" %(tblname,fitname))
    data = c.fetchall()
    data_clean = dict()
    for i in range(len(data)):
        data_clean[data[i][0]] = json.loads(data[i][1])
    return data_clean

def make_ga(data,fitname):
    p = dps.gA_parameters()
    if fitname in ['taylor_esq_1']:
        gA = np.array([ff.ga_epi(epi0=data[0]['e0']**2,epi=p['epi_phys']**2,a=0,**data[i]) for i in range(len(data))])
    elif fitname in ['chiral_nlo']:
        gA = np.array([ff.ga_su2(epi=p['epi_phys'],a=0,**data[i]) for i in range(len(data))])
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

def make_histogram(bssort, title, tag, weights=None, param=None, boot0=None):
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
        CIidx = {0.025:int(norm*0.025), 0.159:int(norm*0.158655254), 0.500:int(norm*0.5), 0.841:int(norm*0.841344746), 0.975:int(norm*0.975)}
    CI = [bssort[CIidx[0.159]], bssort[CIidx[0.841]], bssort[CIidx[0.500]]]
    print "median: %s" %CI[2]
    print "dCI: %s" %(0.5*(CI[1]-CI[0]))
    print "std: %s" %np.std(bssort)

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
    x = np.delete(bins, -1)
    if param==None:
        ax.set_xlabel('$g_{A}$', fontsize=20)
    else:
        ax.set_xlabel('%s' %param, fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=0)
    ax.set_title(title,x=0.15,y=0.68/0.8,fontsize=20,bbox=dict(facecolor=color))
    #plt.suptitle('%s' %ens,x=0.2,y=0.9,fontsize=20)
    #plt.tight_layout()
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    plt.draw()
    plt.show()
    fig.savefig('/Users/cchang5/Documents/Papers/c51_p2/papers/ga_long/gA_%s.pdf' %tag, format='pdf')
    return 0

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
    return w

def model_avg(boot0,bootn,weights):
    cv = 0
    for k in boot0.keys():
        cv += weights[k]*boot0[k]
    sdev = 0
    for k in bootn.keys():
        sdev += weights[k]*np.sqrt(np.var(bootn[k]) + (boot0[k] - cv)**2)
    return cv, sdev

if __name__=='__main__':
    taylor_esq_1 = read_sql('xcont','taylor_esq_1')
    gA_tesq1_0, gA_tesq1_n = make_ga(taylor_esq_1,'taylor_esq_1')
    #make_histogram(gA_tesq1_n,title='Taylor series in $\epsilon_\pi^2$',tag='taylor_esq_1')

    chiral_nlo = read_sql('xcont','chiral_nlo')
    gA_xnlo_0, gA_xnlo_n = make_ga(chiral_nlo,'chiral_nlo')
    #make_histogram(gA_xnlo_n,title='SU(2) NLO',tag='chiral_nlo')

    # model average
    # get Akaike weights
    data0 = dict()
    data0['taylor_esq_1'] = taylor_esq_1[0]
    data0['chiral_nlo'] = chiral_nlo[0]
    w = akaike_weights(data0)
    # average models
    boot0 = dict()
    boot0['taylor_esq_1'] = gA_tesq1_0
    boot0['chiral_nlo'] = gA_xnlo_0
    bootn = dict()
    bootn['taylor_esq_1'] = gA_tesq1_n
    bootn['chiral_nlo'] = gA_xnlo_n
    mean, sdev = model_avg(boot0,bootn,w)
    print mean, sdev
    # average histogram
    whist = np.concatenate((w['taylor_esq_1']*np.ones_like(gA_tesq1_n),w['chiral_nlo']*np.ones_like(gA_xnlo_n)),axis=0)
    cbootn = np.concatenate((gA_tesq1_n, gA_xnlo_n),axis=0)
    idx = np.argsort(cbootn)
    make_histogram(cbootn[idx],title='Akaike average',tag='AIC',weights=whist[idx])
