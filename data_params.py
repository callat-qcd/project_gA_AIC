import numpy as np

def gA_parameters():
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
    p['dbname'] = 'c51_ga.sqlite'
    p['tblname'] = 'xcont'
    return p

def plotting_parameters():
    p = dict()
    # Figure sizes and fonts
    #p['fig_gldn'] = (8.125,5.018)
    p['fig_gldn'] = (7,4.326237)
    #p['ga_axes'] = [0.095,0.128,0.895,0.865]
    #p['mL_axes'] = [0.095,0.125,0.895,0.865]
    p['ga_axes'] = [0.15,0.15,0.8,0.8]
    p['mL_axes'] = [0.15,0.15,0.8,0.8]
    p['fs'] = 20

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


