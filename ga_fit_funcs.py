import sys
import numpy as np
import scipy.special as spsp
import iminuit as mn

# this holds fit functions and chisq for fitting ga

def Q(chisq,dof):
    return spsp.gammaincc(0.5*dof,0.5*chisq)
def aicc(chisq,l_d,l_par):
    return 2*l_par + chisq + 2.*(l_par+1)*(l_par+2)/(l_d -l_par -2)
def minimize(chisq,ini_vals,l_d):
    ga_min = mn.Minuit(chisq, pedantic=False, print_level=0, **ini_vals)
    ga_min.migrad()
    dof = l_d - len(ga_min.values)
    print "chi^2 = %.4f, dof = %d, Q = %.4f" %(ga_min.fval,dof,Q(ga_min.fval,dof))
    for p in ga_min.parameters:
        print '  %s = %.4f +- %.4f' %(p,ga_min.values[p],ga_min.errors[p])
    cov = np.array(ga_min.matrix(correlation=False,skip_fixed=True))
    return ga_min, cov

########################################
#  gA SU(2) ChiPT vs e_pi = mpi / 4piFpi
########################################
def ga_su2(epi,a,g0,c2,c3=0,ca2=0,ca4=0,afs=0,cafs=0):
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
    ga += cafs * a**2 * afs
    ga += ca4 * a**4
    return ga
def dga_su2(epi,a,g0,c2,lam_cov,c3=0,ca2=0,ca4=0,afs=0,cafs=0):
    ln = np.log(epi**2)
    if ca2 == 0 and c3 == 0 and ca4 == 0 and cafs == 0:
        dgdl = np.array([1 - (1 + 6 * g0**2)*epi**2 * ln, epi**2])
    elif c3 == 0 and ca4 == 0 and cafs == 0:
        dgdl = np.array([1 - (1 + 6 * g0**2)*epi**2 * ln, epi**2, a**2])
    elif ca2 == 0 and ca4 == 0 and cafs == 0:
        dgdl = np.array([1 - (1 + 6 * g0**2)*epi**2 * ln + c3*epi**3, epi**2, g0*epi**3])
    elif ca4 == 0 and cafs == 0:
        dgdl = np.array([1 - (1 + 6 * g0**2)*epi**2 * ln + c3*epi**3, epi**2, g0*epi**3, a**2])
    else:
        print('ca4, cafs currently unsupported')
        sys.exit()
    #dgdl = np.array([1 - e_pi * log * (2 + 12*g0),e_pi,0])
    #lam_cov[:,2] = 0.
    #lam_cov[2,:] = 0.
    g_err = np.sqrt(np.dot(dgdl,np.dot(lam_cov,dgdl)))
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

###################################
#  FV FUNCTIONS
###################################
fv_weights = {
    1:6, 2:12, 3:8, 4:6, 5:24, 6:24, 7:0, 8:12, 9:30, 10:24,
    11:24, 12:8, 13:24, 14:48, 15:0, 16:6, 17:48, 18:36, 19:24, 20:24,
    }
cn = np.array([6,12,8,6,24,24,0,12,30,24,24,8,24,48,0,6,48,36,24,24])
n_mag = np.arange(1,len(cn)+1,1)
def ga_f1(mL):
    f1 = np.sum(cn * spsp.kn(0,mL*n_mag))
    f1 -= np.sum(cn * spsp.kn(1,mL*n_mag) / mL / n_mag)
    return f1
def ga_f3(mL):
    f3 = -3./2 * np.sum(cn * spsp.kn(1,mL*n_mag) / mL / n_mag)
    return f3
def dgaFV(epi,mL,g0):
    return 8./3 * epi**2 * (g0**3 * ga_f1(mL) + g0 * ga_f3(mL))
def ddgaFV(epi,mL,g0):
    return 8./3 * epi**2 * (3 * g0**2 * ga_f1(mL) + ga_f3(mL))

###################################
#  TAYLOR EXPANSION
###################################
def ga_epi(epi0,epi,a,c0,ca2=0,cm1=0,cm2=0,cam2=0,**kwargs):
    ga = c0
    if ca2 != 0:
        ga += ca2*a**2
    if cm1 != 0:
        ga += cm1*(epi - epi0)
    if cm2 != 0:
        ga += cm2*(epi - epi0)**2
    if cam2 != 0:
        ga += cam2 * (epi - epi0)**2 * a**2
    return ga
def dga_epi(epi0,epi,a,c0,lam_cov,ca2=0,cm1=0,cm2=0,cam2=0,**kwargs):
    if ca2 == 0 and cm1 == 0 and cm2 == 0 and cam2 == 0:
        dgdl = np.array([np.ones_like(epi)])
    elif ca2 != 0 and cm1 == 0 and cm2 == 0 and cam2 == 0:
        dgdl = np.array([np.ones_like(epi),a**2])
    elif ca2 != 0 and cm1 != 0 and cm2 == 0 and cam2 == 0:
        dgdl = np.array([np.ones_like(epi),epi-epi0,a**2])
    elif ca2 != 0 and cm1 != 0 and cam2 != 0 and cm2 == 0:
        dgdl = np.array([np.ones_like(epi),epi-epi0,a**2,(epi-epi0)*a**2])
    elif ca2 != 0 and cm1 == 0 and cam2 != 0 and cm2 == 0:
        dgdl = np.array([np.ones_like(epi),a**2,(epi-epi0)*a**2])
    else:
        dgdl = np.array([np.ones_like(epi),epi-epi0,(epi-epi0)**2,a**2])
    g_err = np.sqrt(np.dot(dgdl,np.dot(lam_cov,dgdl)))
    return g_err
def dga_a(epi0,epi,a,c0,lam_cov,ca2=0,cm1=0,cm2=0,**kwargs):
    if ca2 == 0 and cm1 == 0 and cm2 == 0:
        dgdl = np.array([np.ones_like(a)])
    elif ca2 != 0 and cm1 == 0 and cm2 == 0:
        dgdl = np.array([np.ones_like(a),a**2])
    elif ca2 != 0 and cm1 != 0 and cm2 == 0:
        dgdl = np.array([np.ones_like(a),np.ones_like(a)*(epi-epi0),a**2])
    else:
        dgdl = np.array([np.ones_like(a),\
            np.ones_like(a)*(epi-epi0),np.ones_like(a)*(epi-epi0)**2,a**2])
    g_err = np.ones_like(a)
    for i,ai in enumerate(a):
        g_err[i] = np.sqrt(np.dot(dgdl[:,i],np.dot(lam_cov,dgdl[:,i])))
    return g_err


