import sys
import numpy as np
import scipy.special as spsp
import iminuit as mn
import time
import sqlite_store as sql
import tqdm

# this holds fit functions and chisq for fitting ga

def Q(chisq,dof):
    return spsp.gammaincc(0.5*dof,0.5*chisq)
def aicc(chisq,l_d,l_par):
    return 2*l_par + chisq + 2.*(l_par)*(l_par+1)/(l_d -l_par -1)
def aic(chisq,l_par):
    return 2*l_par + chisq
def minimize(chisq,ini_vals):
    ga_min = mn.Minuit(chisq, pedantic=False, print_level=0, **ini_vals)
    ga_min.migrad()
    return ga_min

########################################
#  gA SU(2) ChiPT vs e_pi = mpi / 4piFpi
########################################
def ga_su2(epi,a,g0,c2,c3=0,ca2=0,cam2=0,ca4=0,afs=0,cafs=0,**kwargs):
    # LO relations
    ga = g0
    # asq
    ga += ca2 * a**2
    # NLO relations non-analytic + c.t.
    ga += -(g0 + 2*g0**3) * epi**2 * np.log(epi**2)
    ga += c2 * epi**2
    # NNLO relation - only non-analytic
    ga += g0 * c3 * epi**3
    # more continuum extrapolation terms
    ga += cam2 * epi**2 * a**2
    ga += cafs * a**2 * afs
    ga += ca4 * a**4
    return ga
def dga_su2(epi,a,g0,c2,lam_cov,c3=0,ca2=0,cam2=0,ca4=0,afs=0,cafs=0,**kwargs):
    if type(epi) != np.ndarray and type(a) == np.ndarray:
        ones = np.ones_like(a)
    elif type(a) != np.ndarray and type(epi) == np.ndarray:
        ones = np.ones_like(epi)
    else:
        print('a or epi needs to be an int/float and the other is a numpy array')
        raise SystemExit
    ln = np.log(epi**2)
    if ca2 == 0 and c3 == 0 and ca4 == 0 and cafs == 0 and cam2 == 0:
        dgdl = np.array([ones*1 - (1 + 6 * g0**2)*epi**2 * ln, ones*epi**2])
    elif c3 == 0 and ca4 == 0 and cafs == 0 and ca2 != 0 and cam2 != 0:
        dgdl = np.array([ones*1 - (1 + 6 * g0**2)*epi**2 * ln,\
            ones*epi**2,ones*a**2,ones*epi**2*a**2])
    elif c3 == 0 and ca4 == 0 and cafs == 0 and cam2 == 0:
        dgdl = np.array([ones*1 - (1 + 6 * g0**2)*epi**2 * ln, ones*epi**2, ones*a**2])
    elif ca2 == 0 and ca4 == 0 and cafs == 0 and cam2 == 0:
        dgdl = np.array([ones*1 -(1 +6 *g0**2)*epi**2 *ln +c3*epi**3, ones*epi**2, ones*g0*epi**3])
    elif ca4 == 0 and cafs == 0 and cam2 == 0:
        dgdl = np.array([ones*1 -(1 +6 *g0**2)*epi**2 *ln +c3*epi**3, ones*epi**2, ones*g0*epi**3, ones*a**2])
    else:
        print('ca4, cafs currently unsupported')
        sys.exit()
    g_err = ones
    for i in range(len(ones)):
        g_err[i] = np.sqrt(np.dot(dgdl[:,i],np.dot(lam_cov,dgdl[:,i])))
    #g_err = np.sqrt(np.dot(dgdl,np.dot(lam_cov,dgdl)))
    return g_err
def ga_su2_nlo(epi,g0,c2):
    # NLO relations non-analytic + c.t.
    ga = -(g0 + 2*g0**3) * epi**2 * np.log(epi**2)
    ga += c2 * epi**2
    return ga
def ga_su2_nnlo(epi,g0,c3):
    # NNLO relation - only non-analytic
    ga = g0 * c3 * epi**3
    return ga
def ga_ma_nlo(epi,eju,epqsq,a,g0,g0b,c2,ca2,afs=0,cafs=0,**kwargs):
    ga   = g0
    ga  += -(g0 + 2*g0**3) * epi**2 * np.log(epi**2)
    ga  += c2 * epi**2
    g0ma = g0 +(24.*g0**3 -15*g0**2*g0b+14*g0*g0b**2+g0b**3)/12
    ga  += -g0ma*(eju**2*np.log(eju**2) -epi**2*np.log(epi**2))
    ga  += -g0*g0b**2 *epqsq * (1+np.log(epi**2))
    ga  += ca2 * a**2
    return ga


###################################
#  FV FUNCTIONS
###################################
class FV_function():
    def __init__(self,epi,mL):
        self.epi = np.array(epi)
        self.mL = np.array(mL)

        self.fv_weights = {
            1:6, 2:12, 3:8, 4:6, 5:24, 6:24, 7:0, 8:12, 9:30, 10:24,
            11:24, 12:8, 13:24, 14:48, 15:0, 16:6, 17:48, 18:36, 19:24, 20:24,
            }
        self.cn = np.array([6,12,8,6,24,24,0,12,30,24,24,8,24,48,0,6,48,36,24,24])
        self.n_mag = np.arange(1,len(self.cn)+1,1)
        self.cn_matrix = np.zeros(self.mL.shape+(1,)) + self.cn
        self.n_mat = np.zeros(self.mL.shape+(1,)) + self.n_mag
        # hacky - hacky - fix cause of STUPID BEHAVIOR OF np.rollaxis
        self.mL_mat = np.rollaxis(np.array([self.mL for i in range(len(self.cn))]).T,-1).T
        # define look up tables of BesselK(n,z) functions cause they are slow
        self.kn0 = spsp.kn(0,self.mL_mat * self.n_mat)
        self.kn1 = spsp.kn(1,self.mL_mat * self.n_mat)

        return None

    def ga_f1(self):
        f1  = np.sum(self.cn_matrix * self.kn0,axis=-1)
        f1 -= np.sum(self.cn_matrix * self.kn1/ self.mL_mat / self.n_mat,axis=-1)
        return f1
    def ga_f3(self):
        f3  = -3./2 *np.sum(self.cn_matrix *self.kn1 / self.mL_mat / self.n_mat ,axis=-1)
        return f3
    def dgaFV(self,g0):
        f1 = self.ga_f1()
        f3 = self.ga_f3()
        return 8./3 * self.epi**2 * (g0**3 * f1 + g0 * f3)
    def ddgaFV(self,g0):
        return 8./3 * self.epi**2 * (3 * g0**2 * self.ga_f1() + self.ga_f3())

def dfv_su2_nlo(epi,mL,a,g0,c2,ca2,lam_cov,cam2=0,**kwargs):
    fv_class = FV_function(epi,mL)
    dfv = fv_class.dgaFV(g0)
    if cam2 == 0:
        dgdl = np.array([
            1 - (1 + 6 * g0**2)*epi**2 * np.log(epi**2) + dfv,\
            epi**2,\
            a**2])
    else:
        dgdl = np.array([
            1 - (1 + 6 * g0**2)*epi**2 * np.log(epi**2) + dfv,\
            epi**2,\
            a**2,a**2 * epi**2])        
    return np.sqrt(np.dot(dgdl,np.dot(lam_cov,dgdl)))


###################################
#  TAYLOR EXPANSION
###################################
def ga_epi(epi0,epi,a,c0,ca2=0,cm1=0,cm2=0,cam2=0,**kwargs):
    ga  = c0
    ga += ca2*a**2
    ga += cm1*(epi - epi0)
    ga += cm2*(epi - epi0)**2
    ga += cam2 * (epi - epi0) * a**2
    return ga
def dga_epi(epi0,epi,a,c0,lam_cov,ca2=0,cm1=0,cm2=0,cam2=0,**kwargs):
    if type(epi) != np.ndarray and type(a) == np.ndarray:
        ones = np.ones_like(a)
    elif type(a) != np.ndarray and type(epi) == np.ndarray:
        ones = np.ones_like(epi)
    else:
        print('a or epi needs to be an int/float and the other is a numpy array')
        raise SystemExit
    if ca2 == 0 and cm1 == 0 and cm2 == 0 and cam2 == 0:
        dgdl = np.array([ones])
    elif ca2 != 0 and cm1 == 0 and cm2 == 0 and cam2 == 0:
        dgdl = np.array([ones,ones*a**2])
    elif ca2 == 0 and cm1 != 0 and cm2 == 0 and cam2 == 0:
        dgdl = np.array([ones,ones*epi-epi0])
    elif ca2 != 0 and cm1 != 0 and cm2 == 0 and cam2 == 0:
        dgdl = np.array([ones,ones*epi-epi0,ones*a**2])
    elif ca2 != 0 and cm1 != 0 and cam2 != 0 and cm2 == 0:
        dgdl = np.array([ones,ones*epi-epi0,ones*a**2,ones*(epi-epi0)*a**2])
    elif ca2 != 0 and cm1 == 0 and cam2 != 0 and cm2 == 0:
        dgdl = np.array([ones,ones*a**2,ones*(epi-epi0)*a**2])
    else:
        dgdl = np.array([ones,ones*epi-epi0,ones*(epi-epi0)**2,ones*a**2])
    g_err = ones
    for i in range(len(ones)):
        g_err[i] = np.sqrt(np.dot(dgdl[:,i],np.dot(lam_cov,dgdl[:,i])))
    return g_err
def dga_epi_fv(epi0,epi,epifv,a,mL,lam_cov,c0,g0fv,ca2=0,cm1=0,cm2=0,cam2=0,**kwargs):
    # note taylor expansion epi can be episq if that is the expansion parameter
    # and not necessarily the same as epi
    fv_class = FV_function(epifv,mL)
    dfv = fv_class.dgaFV(g0fv)
    #dfv = 8./3 * epifv**2 * (3 * g0fv**2 * ga_f1(mL) + ga_f3(mL))
    if   ca2 == 0 and cm1 == 0 and cm2 == 0 and cam2 == 0:
        dgdl = np.array([1,dfv])
    elif ca2 != 0 and cm1 == 0 and cm2 == 0 and cam2 == 0:
        dgdl = np.array([1,a**2,dfv])
    elif ca2 == 0 and cm1 != 0 and cm2 == 0 and cam2 == 0:
        dgdl = np.array([1,epi**2,dfv])
    elif ca2 != 0 and cm1 != 0 and cm2 == 0 and cam2 == 0:
        dgdl = np.array([1,epi-epi0,a**2,dfv])
    elif ca2 != 0 and cm1 != 0 and cm2 == 0 and cam2 != 0:
        dgdl = np.array([1,epi-epi0,a**2,(epi-epi0)*a**2,dfv])
    elif ca2 != 0 and cm1 != 0 and cm2 != 0 and cam2 == 0:
        dgdl = np.array([1,epi-epi0,(epi-epi0)**2,a**2,dfv])
    elif ca2 != 0 and cm1 != 0 and cm2 != 0 and cam2 != 0:
        dgdl = np.array([1,epi-epi0,(epi-epi0)**2,a**2,(epi-epi0)*a**2,dfv])
    else:
        print('unrecognized set of parameter options for dga_epi_fv')
        print('c0:',c0,'g0fv:',g0fv,'ca2:',ca2,'cm1:',cm1,'cm2:',cm2,'cam2:',cam2)
        raise SystemExit
    g_err = np.sqrt(np.dot(dgdl,np.dot(lam_cov,dgdl)))
    return g_err

class ChiSq():
    def __init__(self,args,p,data,select):
        self.select = select
        self.args = args
        self.p = p
        self.afs = np.zeros(len(self.p['afs']))
        for e in self.p['ens_idx']:
            self.afs[self.p['ens_idx'][e]] = self.p['afs'][e]
        self.ga_bs = data['ga_bs']
        self.ga_b0 = data['ga_b0']
        self.epi_b0 = data['epi_b0']
        self.mL_b0 = data['mL_b0']
        self.aw0_b0 = data['aw0_b0']
        self.aSaw0_b0 = data['aSaw0_b0']
        self.eju_b0 = data['eju_b0']
        self.epqsq_b0 = data['epqsq_b0']
        if not args.error_epi:
            self.epi_bs = data['epi_bs'].mean(axis=0) + np.zeros_like(data['epi_bs'])
            self.eju_bs = data['eju_bs'].mean(axis=0) + np.zeros_like(data['eju_bs'])
        else:
            self.epi_bs = data['epi_bs']
            self.eju_bs = data['eju_bs']
        if not args.error_mL:
            self.mL_bs = data['mL_bs'].mean(axis=0) + np.zeros_like(data['mL_bs'])
        else:
            self.mL_bs = data['mL_bs']
        if not args.error_a:
            self.aw0_bs = data['aw0_bs'].mean(axis=0) + np.zeros_like(data['aw0_bs'])
            self.aSaw0_bs = data['aSaw0_bs'].mean(axis=0) + np.zeros_like(data['aSaw0_bs'])
            self.epqsq_bs = data['epqsq_bs'].mean(axis=0) + np.zeros_like(data['epqsq_bs'])
        else:
            self.aw0_bs = data['aw0_bs']
            self.epqsq_bs = data['epqsq_bs']
            self.aSaw0_bs = data['aSaw0_bs']
        self.do_bs = False
        self.FV_class = FV_function(self.epi_b0,self.mL_b0)
        self.FV_class_bs = FV_function(self.epi_bs,self.mL_bs)
        return None
    def __call__(self,do_bs,bs):
        self.do_bs = do_bs
        self.bs = bs
    def select_chisq(self):
        if self.select in ['t_esq1_a2','t_esq1_aSa2']:
            return self.t_esq1_a2
        elif self.select in ['t_esq0_a0']:
            return self.t_esq0_a0
        elif self.select in ['t_esq0_a2','t_esq0_aSa2']:
            return self.t_esq0_a2
        elif self.select in ['t_esq1_a0']:
            return self.t_esq1_a0
        elif self.select in ['x_nlo_a2','x_nlo_aSa2']:
            return self.x_nlo_a2
        elif self.select in ['x_nlo_a0']:
            return self.x_nlo_a0
        elif self.select in ['c0_nofv']:
            return self.c0_nofv
        elif self.select in ['xma_nlo_a2','xma_nlo_aSa2']:
            return self.xma_nlo_a2
        elif self.select in ['x_nlo_a2_ea2']:
            return self.x_nlo_a2_ea2
        elif self.select in ['t_esq1_a2_ea2']:
            return self.t_esq1_a2_ea2
        else:
            print('chisq is unselected')
            raise SystemExit
    def set_xy(self):
        if self.do_bs:
            self.y = self.ga_bs[self.bs]
            self.xju   = self.eju_bs[self.bs]
            self.epqsq = self.epqsq_bs[self.bs]
            if 'esq' in self.select:
                self.x = (self.epi_bs**2)[self.bs]
            else:
                self.x = self.epi_bs[self.bs]
            if 'aSa2' in self.select:
                self.xa = self.aw0_bs[self.bs] * np.sqrt(self.afs)
            else:
                self.xa = self.aw0_bs[self.bs]
        else:
            self.y = self.ga_b0
            self.xju   = self.eju_b0
            self.epqsq = self.epqsq_b0
            if 'esq' in self.select:
                self.x = self.epi_b0**2
            else:
                self.x = self.epi_b0
            if 'aSa2' in self.select:
                self.xa = self.aw0_b0 * np.sqrt(self.afs)
            else:
                self.xa = self.aw0_b0
        if 'aSa2' in self.select:
            self.xabs = self.aw0_bs * np.sqrt(self.afs)
        else:
            self.xabs = self.aw0_bs
        return 0
        
    def t_esq1_a2(self,c0,cm1,ca2,g0fv):
        ''' taylor esq function defined as function of epi**2 '''
        self.x0 = self.args.e0**2
        self.xphys = self.p['epi_phys']**2
        self.xdict = {'epi0':self.x0, 'epi':self.xphys}
        self.set_xy()

        cdict = {'c0':c0, 'ca2':ca2, 'cm1':cm1} 
        f  = ga_epi(epi0=self.x0,epi=self.x,a=self.xa,**cdict)
        f += self.FV_class.dgaFV(g0fv)
        # IF error_x = True, construct covariance from ybs - f(xbs)
        # where some of the x-errors may have been turned off
        # ELSE
        #   construct simple covariance from only gA_bs
        if self.args.error_x:
            fbs  = ga_epi(self.x0,(self.epi_bs**2),a=xabs,**cdict)
            fbs += self.FV_class_bs.dgaFV(g0fv)
            cov  = np.var( self.ga_bs - fbs,axis=0)
        else:
            cov = self.ga_bs.var(axis=0)
        '''
        we have no PQ data on same ensembles so
        y,f,cov are all len(l_d) arrays
        numpy will properly do the multiplication/division
        '''
        chisq = np.sum( (self.y-f)**2 / cov )
        if self.args.g0fv != None:
            chisq += (g0fv - self.args.g0fv[0])**2 / self.args.g0fv[1]**2
        return chisq
    def x_nlo_a2(self,g0,c2,ca2):
        ''' chipt function defined as function of epi '''
        self.xphys = self.p['epi_phys']
        self.xdict = {'epi':self.xphys}
        self.set_xy()

        cdict = {'g0':g0, 'ca2':ca2, 'c2':c2}
        f     = ga_su2(epi=self.x,a=self.xa,**cdict)
        f    += self.FV_class.dgaFV(g0)
        if self.args.error_x:
            fbs  = ga_su2(epi=self.epi_bs,a=self.xabs,**cdict)
            fbs += self.FV_class_bs.dgaFV(g0)
            cov  = np.var( self.ga_bs - fbs,axis=0)
        else:
            cov = self.ga_bs.var(axis=0)
        chisq = np.sum( (self.y-f)**2 / cov )
        return chisq
    def x_nlo_a0(self,g0,c2):
        ''' chipt function defined as function of epi '''
        self.xphys = self.p['epi_phys']
        self.xdict = {'epi':self.xphys}
        self.set_xy()
        cdict = {'g0':g0,'c2':c2}
        f  = ga_su2(epi=self.x,a=self.xa,**cdict)
        f += self.FV_class.dgaFV(g0)
        if self.args.error_x:
            fbs  = ga_su2(epi=self.epi_bs,a=self.xabs,**cdict)
            fbs += self.FV_class_bs.dgaFV(g0)
            cov  = np.var( self.ga_bs - fbs,axis=0)
        else:
            cov = self.ga_bs.var(axis=0)
        chisq = np.sum( (self.y-f)**2 / cov )
        return chisq
    def t_esq0_a2(self,c0,ca2,g0fv):
        ''' taylor esq function defined as function of epi**2 '''
        self.x0 = self.args.e0**2
        self.xphys = self.p['epi_phys']**2
        self.xdict = {'epi0':self.x0, 'epi':self.xphys}
        self.set_xy()
        cdict = {'c0':c0, 'ca2':ca2}
        f  = ga_epi(epi0=self.x0,epi=self.x,a=self.xa,**cdict)
        f += self.FV_class.dgaFV(g0fv)
        if self.args.error_x:
            fbs  = ga_epi(self.x0,(self.epi_bs**2),self.xabs,**cdict)
            fbs += self.FV_class_bs.dgaFV(g0fv)
            cov  = np.var( self.ga_bs - fbs,axis=0)
        else:
            cov = self.ga_bs.var(axis=0)
        chisq = np.sum( (self.y-f)**2 / cov )
        if self.args.g0fv != None:
            chisq += (g0fv - self.args.g0fv[0])**2 / self.args.g0fv[1]**2
        return chisq
    def t_esq0_a0(self,c0,g0fv):
        ''' taylor esq function defined as function of epi**2 '''
        self.x0 = self.args.e0**2
        self.xphys = self.p['epi_phys']**2
        self.xdict = {'epi0':self.x0, 'epi':self.xphys}
        self.set_xy()
        cdict = {'c0':c0}
        f  = ga_epi(epi0=self.x0,epi=self.x,a=self.xa,**cdict)
        f += self.FV_class.dgaFV(g0fv)
        if self.args.error_x:
            fbs  = ga_epi(self.x0,(self.epi_bs**2),self.xabs,**cdict)
            fbs += self.FV_class_bs.dgaFV(g0fv)
            cov  = np.var( self.ga_bs - fbs,axis=0)
        else:
            cov = self.ga_bs.var(axis=0)
        chisq = np.sum( (self.y-f)**2 / cov )
        if self.args.g0fv != None:
            chisq += (g0fv - self.args.g0fv[0])**2 / self.args.g0fv[1]**2
        return chisq
    def c0_nofv(self,c0):
        ''' taylor esq function defined as function of epi**2 '''
        self.x0 = self.args.e0**2
        self.xphys = self.p['epi_phys']**2
        self.xdict = {'epi0':self.x0, 'epi':self.xphys}
        self.set_xy()
        cdict = {'c0':c0}
        f  = ga_epi(epi0=self.x0,epi=self.x,a=0,**cdict)
        if self.args.error_x:
            fbs  = ga_epi(epi0=0,epi=(self.epi_bs**2),a=0,**cdict)
            cov  = np.var( self.ga_bs - fbs,axis=0)
        else:
            cov = self.ga_bs.var(axis=0)
        chisq = np.sum( (self.y-f)**2 / cov )
        return chisq
    def t_esq1_a0(self,c0,cm1,g0fv):
        ''' taylor esq function defined as function of epi**2 '''
        self.x0 = self.args.e0**2
        self.xphys = self.p['epi_phys']**2
        self.xdict = {'epi0':self.x0, 'epi':self.xphys}
        self.set_xy()
        cdict = {'c0':c0, 'cm1':cm1}
        f  = ga_epi(epi0=self.x0,epi=self.x,a=self.xa,**cdict)
        f += self.FV_class.dgaFV(g0fv)
        if self.args.error_x:
            fbs  = ga_epi(epi0=self.x0,epi=(self.epi_bs**2),a=self.xabs,**cdict)
            fbs += self.FV_class_bs.dgaFV(g0fv)
            cov  = np.var( self.ga_bs - fbs,axis=0)
        else:
            cov = self.ga_bs.var(axis=0)
        chisq = np.sum( (self.y-f)**2 / cov )
        if self.args.g0fv != None:
            chisq += (g0fv - self.args.g0fv[0])**2 / self.args.g0fv[1]**2
        return chisq
    def x_nlo_a2_ea2(self,g0,c2,ca2,cam2):
        ''' chipt function defined as function of epi '''
        self.xphys = self.p['epi_phys']
        self.xdict = {'epi':self.xphys}
        self.set_xy()
        cdict = {'g0':g0, 'ca2':ca2, 'c2':c2, 'cam2':cam2}
        f  = ga_su2(epi=self.x,a=self.xa,**cdict)
        f += self.FV_class.dgaFV(g0)
        if self.args.error_x:
            fbs  = ga_su2(epi=self.epi_bs,a=self.xabs,**cdict)
            fbs += self.FV_class_bs.dgaFV(g0)
            cov  = np.var( self.ga_bs - fbs,axis=0)
        else:
            cov = self.ga_bs.var(axis=0)
        chisq = np.sum( (self.y-f)**2 / cov )
        return chisq
    def t_esq1_a2_ea2(self,c0,cm1,ca2,cam2,g0fv):
        ''' taylor esq function defined as function of epi**2 '''
        self.x0 = self.args.e0**2
        self.xphys = self.p['epi_phys']**2
        self.xdict = {'epi0':self.x0, 'epi':self.xphys}
        self.set_xy()
        cdict = {'c0':c0, 'ca2':ca2, 'cm1':cm1, 'cam2':cam2} 
        f     = ga_epi(epi0=self.x0,epi=self.x,a=self.xa,**cdict)
        f    += self.FV_class.dgaFV(g0fv)
        if self.args.error_x:
            fbs  = ga_epi(self.x0,(self.epi_bs**2),a=self.xabs,**cdict)
            fbs += self.FV_class_bs.dgaFV(g0fv)
            cov  = np.var( self.ga_bs - fbs,axis=0)
        else:
            cov = self.ga_bs.var(axis=0)
        chisq = np.sum( (self.y-f)**2 / cov )
        if self.args.g0fv != None:
            chisq += (g0fv - self.args.g0fv[0])**2 / self.args.g0fv[1]**2
        return chisq
    def xma_nlo_a2(self,g0,c2,ca2,g0b):
        ''' chipt function defined as function of epi '''
        self.xphys = self.p['epi_phys']
        self.xdict = {'epi':self.xphys}
        self.set_xy()
        cdict = {'g0':g0, 'c2':c2, 'g0b':g0b, 'ca2':ca2}
        f  = ga_ma_nlo(epi=self.x,eju=self.xju,epqsq=self.epqsq,a=self.xa,**cdict)
        f += self.FV_class.dgaFV(g0)
        if self.args.error_x: #ga_ma_nlo(epi,eju,epqsq,a,g0,g0b,c2,afs=0,cafs=0,**kwargs)
            fbs  = ga_ma_nlo(\
                epi=self.epi_bs,eju=self.eju_bs,epqsq=self.epqsq_bs,a=self.xabs,**cdict)
            fbs += self.FV_class_bs.dgaFV(g0)
            cov  = np.var( self.ga_bs - fbs,axis=0)
        else:
            cov = self.ga_bs.var(axis=0)
        chisq = np.sum( (self.y-f)**2 / cov )
        if self.args.g0b != None:
            chisq += (g0b - self.args.g0b[0])**2 / self.args.g0b[1]**2
        return chisq

def fit_gA(args,p,data,ini_vals):
    def print_output(CS,ga_min,select):
        dof = CS.p['l_d'] - len(ga_min.values)
        print "chi^2 = %.4f, dof = %d, Q = %.4f" %(ga_min.fval,dof,Q(ga_min.fval,dof))
        for i in ga_min.parameters:
            print '  %s = %.4f +- %.4f' %(i,ga_min.values[i],ga_min.errors[i])

        # central value
        xphys = CS.xphys
        # uncertainty - gA-infinite doesn't know about FV
        # so chop covariance matrix - g0fv is last parameter for Taylor fits
        cov = np.array(ga_min.matrix(correlation=False,skip_fixed=True))
        params = CS.xdict.copy()
        params.update(ga_min.values)
        if select in ['t_esq0_a0','t_esq1_a2','t_esq0_a2','t_esq1_a0','t_esq1_a2_ea2',\
            't_esq0_aSa2','t_esq1_aSa2p','t_esq1_aSa2']:
            cov2 = cov[:-1,:-1]
            x0 = CS.x0
            ga_fit = ga_epi(a=0,**params)
            dga_fit = dga_epi(epi0=x0,epi=np.array([xphys]),a=0,lam_cov=cov2,**ga_min.values)
        elif select in ['x_nlo_a0','x_nlo_a2','x_nlo_aSa2','x_nlo_a2_ea2']:
            ga_fit = ga_su2(a=0,**params)
            dga_fit = dga_su2(epi=np.array([xphys]),a=0,lam_cov=cov,**ga_min.values)
        elif select in ['c0_nofv']:
            ga_fit = ga_epi(a=0,**params)
            dga_fit = dga_epi(epi0=0,epi=np.array([xphys]),a=0,lam_cov=cov,**ga_min.values)
        elif select in ['xma_nlo_a2','xma_nlo_aSa2']:
            cov2 = cov[0:-1,0:-1]
            ga_fit = ga_su2(a=0,**params)
            dga_fit = dga_su2(epi=np.array([xphys]),a=0,lam_cov=cov2,**ga_min.values)
        else:
            print(select+' not added to print output')
        print('gA = %.7f +- %.7f' %(ga_fit,dga_fit))
        if 'g0fv' in ga_min.values:
            print('g0fv = %.3f +- %.3f' %(ga_min.values['g0fv'],ga_min.errors['g0fv']))
            if args.g0fv != None:
                print('g0fv prior = %f +- %f' %(args.g0fv[0],args.g0fv[1]))
        print('AIC = 2k - 2 ln(exp(-chisq/2))')
        print('AIC = %.4f\n' %aic(ga_min.fval,len(ga_min.values)))
        return {'ga_fit':ga_fit, 'dga_fit':dga_fit, 'xdict':dict(CS.xdict), 'ga_min':ga_min}
    # record b0 and bs results to DB
    def bs_to_db(p,ga_min,select):
        print('make sqlite db and table')
        cur,conn = sql.login(p)
        cur,conn = sql.id_name_nbs_result_table(cur,conn,p)
        # write boot0
        b0result = dict(ga_min.values)
        b0result['cov'] = np.array(ga_min.matrix(correlation=False,skip_fixed=True)).tolist()
        b0result['AIC'] = aic(ga_min.fval,len(ga_min.values))
        b0result['chi2'] = ga_min.fval
        b0result['dof'] = CS.p['l_d'] - len(ga_min.values)
        b0result['Q'] = Q(ga_min.fval,b0result['dof'])
        b0result['e0'] = args.e0
        b0result = str(b0result).replace("'",'\"')
        cur,conn = sql.id_name_nbs_result_insert(cur,conn,p,select,0,b0result)
        for bs in tqdm.tqdm(range(p['Nbs']),desc='Nbs'):
            CS(True,bs)
            ga_min_bs = minimize(CS.select_chisq(),ini_vals(select))
            cur,conn = sql.id_name_nbs_result_insert(\
                cur,conn,p,select,bs+1,str(ga_min_bs.values).replace("'",'\"'))

    # collect result
    rdict = dict()
    # choose fit function
    if args.fits in ['all','t_esq0_a2']:
        select = 't_esq0_a2'
        CS = ChiSq(args,p,data,select)
        print('==================================================')
        print('gA = c0 + ca2 * (a/w0)**2 + FV\n')
        # do the minimization
        ga_min = minimize(CS.select_chisq(),ini_vals(select))
        # print outputs
        rdict[select] = print_output(CS,ga_min,select)
        if args.bs:
            bs_to_db(p,ga_min,select)
    if args.fits in ['all','t_esq0_aSa2']:
        select = 't_esq0_aSa2'
        CS = ChiSq(args,p,data,select)
        print('==================================================')
        print('gA = c0 + ca2 * alphaS (a/w0)**2 + FV\n')
        # do the minimization
        ga_min = minimize(CS.select_chisq(),ini_vals(select))
        # print outputs
        rdict[select] = print_output(CS,ga_min,select)
        if args.bs:
            bs_to_db(p,ga_min,select)
    if args.fits in ['all','t_esq1_a0']:
        select = 't_esq1_a0'
        CS = ChiSq(args,p,data,select)
        print('==================================================')
        print('gA = c0 + c1*(epi**2-e0**2) + FV\n')
        # do the minimization
        ga_min = minimize(CS.select_chisq(),ini_vals(select))
        # print outputs
        rdict[select] = print_output(CS,ga_min,select)
        if args.bs:
            bs_to_db(p,ga_min,select)
    if args.fits in ['all','t_esq1_a2']:
        select = 't_esq1_a2'
        CS = ChiSq(args,p,data,select)
        print('==================================================')
        print('gA = c0 + c1*(epi**2-e0**2) + ca2 * (a/w0)**2 + FV\n')
        # do the minimization
        ga_min = minimize(CS.select_chisq(),ini_vals(select))
        # print outputs
        rdict[select] = print_output(CS,ga_min,select)
        if args.bs:
            bs_to_db(p,ga_min,select)
    if args.fits in ['all','t_esq1_aSa2']:
        select = 't_esq1_aSa2'
        CS = ChiSq(args,p,data,select)
        print('==================================================')
        print(select)
        print('gA = c0 + c1*(epi**2-e0**2) + ca2 * alphaS (a/w0)**2 + FV\n')
        # do the minimization
        ga_min = minimize(CS.select_chisq(),ini_vals(select))
        # print outputs
        rdict[select] = print_output(CS,ga_min,select)
        if args.bs:
            bs_to_db(p,ga_min,select)
    if args.fits in ['all','x_nlo_a0']:
        select = 'x_nlo_a0'
        CS = ChiSq(args,p,data,select)
        print('==================================================')
        print('gA = NLO SU(2) + FV, g0fv == g0\n')
        # do the minimization
        ga_min = minimize(CS.select_chisq(),ini_vals(select))
        # print outputs
        rdict[select] = print_output(CS,ga_min,select)
        if args.bs:
            bs_to_db(p,ga_min,select)
    if args.fits in ['all','x_nlo_a2']:
        select = 'x_nlo_a2'
        CS = ChiSq(args,p,data,select)
        print('==================================================')
        print('gA = NLO SU(2) + FV + a**2, g0fv == g0\n')
        # do the minimization
        ga_min = minimize(CS.select_chisq(),ini_vals(select))
        # print outputs
        rdict[select] = print_output(CS,ga_min,select)
        if args.bs:
            bs_to_db(p,ga_min,select)
    if args.fits in ['all','x_nlo_aSa2']:
        select = 'x_nlo_aSa2'
        CS = ChiSq(args,p,data,select)
        print('==================================================')
        print('gA = NLO SU(2) + FV + alphaS * a**2, g0fv == g0\n')
        # do the minimization
        ga_min = minimize(CS.select_chisq(),ini_vals(select))
        # print outputs
        rdict[select] = print_output(CS,ga_min,select)
        if args.bs:
            bs_to_db(p,ga_min,select)
    if args.fits in ['other','c0_nofv']:
        select = 'c0_nofv'
        CS = ChiSq(args,p,data,select)
        print('==================================================')
        print('gA = c0\n')
        # do the minimization
        ga_min = minimize(CS.select_chisq(),ini_vals(select))
        # print outputs
        rdict[select] = print_output(CS,ga_min,select)
        if args.bs:
            bs_to_db(p,ga_min,select)
    if args.fits in ['other','t_esq0_a0']:
        select = 't_esq0_a0'
        CS = ChiSq(args,p,data,select)
        print('==================================================')
        print('gA = c0 + FV\n')
        # do the minimization
        ga_min = minimize(CS.select_chisq(),ini_vals(select))
        # print outputs
        rdict[select] = print_output(CS,ga_min,select)
        if args.bs:
            bs_to_db(p,ga_min,select)
    if args.fits in ['other','x_nlo_a2_ea2']:
        select = 'x_nlo_a2_ea2'
        CS = ChiSq(args,p,data,select)
        print('==================================================')
        print('gA = NLO SU(2) + FV + a**2 + epi**2 * a**2, g0fv == g0\n')
        # do the minimization
        ga_min = minimize(CS.select_chisq(),ini_vals(select))
        # print outputs
        rdict[select] = print_output(CS,ga_min,select)
        if args.bs:
            bs_to_db(p,ga_min,select)
    if args.fits in ['other','t_esq1_a2_ea2']:
        select = 't_esq1_a2_ea2'
        CS = ChiSq(args,p,data,select)
        print('==================================================')
        print('gA = c0 + c1*(epi**2-e0**2) + ca2 * (a/w0)**2 +cam2 (epi**2-e0**2)*(a/w0)**2FV\n')
        # do the minimization
        ga_min = minimize(CS.select_chisq(),ini_vals(select))
        # print outputs
        rdict[select] = print_output(CS,ga_min,select)
        if args.bs:
            bs_to_db(p,ga_min,select)
    if args.fits in ['other','xma_nlo_a2']:# taken out of all cause not constrained by data
        select = 'xma_nlo_a2'
        CS = ChiSq(args,p,data,select)
        print('==================================================')
        print('gA = MA NLO SU(2) + FV, g0fv == g0\n')
        # do the minimization
        ga_min = minimize(CS.select_chisq(),ini_vals(select))
        # print outputs
        rdict[select] = print_output(CS,ga_min,select)
        if args.bs:
            bs_to_db(p,ga_min,select)
    if args.fits in ['other','xma_nlo_aSa2']:# taken out of all cause not constrained by data
        select = 'xma_nlo_aSa2'
        CS = ChiSq(args,p,data,select)
        print('==================================================')
        print('gA = MA NLO SU(2) + FV, g0fv == g0\n')
        # do the minimization
        ga_min = minimize(CS.select_chisq(),ini_vals(select))
        # print outputs
        rdict[select] = print_output(CS,ga_min,select)
        if args.bs:
            bs_to_db(p,ga_min,select)
    return rdict

