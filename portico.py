#!/usr/bin/python3
'''
*---------------------------------------*
| Program    : portico                  |
| Last Update: 2026/10/08 (YYYY/MM/DD)  |
| Author     : David Ferro-Costas       |
| ORCID      : 0000-0002-8365-4047      |
| License    : MIT                      |
*---------------------------------------*

PORTICO: Projection Onto Roto-Translational and Internal COordinates

Implementation of the internal-coordinate projection algorithm for the
asymptotic classification of transition-state (TS) normal modes for a
dissociation channel:

        R -> TS -> P1 + P2

'''
#=====================================================================#
PROGNAME = "portico"
VERSION  = "v2026.1.4"
#=====================================================================#
HEADER = r'''
+------------------------------------+
|                   _   _            |
|  _ __   ___  _ __| |_(_) ___ ___   |
| | '_ \ / _ \| '__| __| |/ __/ _ \  |
| | |_) | (_) | |  | |_| | (_| (_) | |
| | .__/ \___/|_|   \__|_|\___\___/  |
| |_|                                |
|                                    |
| Projection Onto Roto-Translational |
|     and Internal COordinates       |
+------------------------------------+
'''
#=====================================================================#
import importlib
import sys
#---------------------------------------------------------------------#
def check_libraries():

    required_libraries = {
        "cmath": None,
        "os": None,
        "random": None,
        "time": None,
        "matplotlib": "matplotlib",
        "numpy": "numpy",
        "scipy.linalg": "scipy",
    }

    missing_packages = []

    for module_name, package_name in required_libraries.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_packages.append(package_name or module_name)

    if missing_packages:
        print("ERROR: Missing required Python libraries:")
        for package in missing_packages:
            print(f"  - {package}")
        sys.exit(1)
#---------------------------------------------------------------------#
check_libraries()
#=====================================================================#
HELP = f"""
=======================================================================
 {PROGNAME.upper()} - Projection Onto Roto-Translational and Internal COordinates
=======================================================================

 Usage:
    python3 {PROGNAME}.py fname
    python3 {PROGNAME}.py -g | --ginput
    python3 {PROGNAME}.py -h | --help
    python3 {PROGNAME}.py -v | --version

 Arguments:
    fname             input file

 Options:

    -g, --ginput      generate an example input file

    -v, --version     show the program version

    -h, --help        show this help message and exit

 Examples:
    python3 {PROGNAME}.py portico.inp
    python3 {PROGNAME}.py -g
=======================================================================
"""
#---------------------------------------------------------------------#
INPUTEXAMPLE = '''# Files with the electronic-structure data (gts format)
file_divsurf  DS.gts        # dividing-surface structure (stationary or non-stationary)
file_product1 P1.gts        # product fragment 1
file_product2 P2.gts        # product fragment 2

# Atom mapping (1-based): each atom of a product is assigned to one
# atom of the dividing surface structure ; every atom must appear exactly once
product1:
  1 1
  2 2
  3 3
  4 4
end

product2:
  1 5
  2 6
end

#-------------------------------------------------------------------#
#                        Optional keywords                          #
#        (all of them are optional; default values are shown)       #
#-------------------------------------------------------------------#
file_plot       omegas.png    # name of the plot file
eps_conn        1.30          # scale factor for bonds (connectivity)
eps_ccic        6.00          # max. |freq(cc) - freq(ic)| (cm-1)
linear_cutoff   170.0         # degrees; bond angles >= this value are treated as linear 
stationary      yes           # stationary point? yes/no (no: MEP point)
#seed           9876543210    # random seed; default: taken from the time
#                             # (printed in the output); fix it to repeat a run
#-------------------------------------------------------------------#

'''
#=====================================================================#
import cmath
import os
import random
import time
#---------------------------------------------------------------------#
import matplotlib.pyplot as plt
import numpy             as np
import scipy
#=====================================================================#
CONNECTSCAL    = 1.3
EPS_CCIC       = 6.0 # cm^-1
EPS_FLOAT      = 1e-8
EPS_GIV        = 1e-10
EPS_ICF        = 0.3
EPS_INERTIA    = 1e-7
EPS_NORM       = 1e-7
EPS_SCX        = 1e-7
EPS_SVD        = 1e-9
#----------------------------------------------------------------------#
BOOL_LINEAR    = False
CUTOFF_LINEAR  = 170.0 # linear angle if angle > CUTOFF_LINEAR (degrees)
if BOOL_LINEAR:
   EPS_LINEAR  = 180.0 - CUTOFF_LINEAR
else:
   EPS_LINEAR  = 4.5
#=====================================================================#
#------------- related to radians -------------#
TWOPI    = 6.283185307179586                   #
#---------- physical constants  (SI) ----------#
C0_SI    = 2.99792458E8                        # speed of light  [m/s]
KB_SI    = 1.3806504E-23                       # Boltzmann const [J/K]
ME_SI    = 9.10938356E-031                     # mass electron   [kg]
MC12_SI  = 1.66053904E-027                     # mass C12        [kg]
#-------------- conversion to au --------------#
JOULE    = 4.35974465E-018                     #
METER    = 0.52917721067E-10                   #
SECOND   = 1.0 / 4.1341373337e+16              #
ANGSTROM = METER * 1E10                        #
AMU      = ME_SI/MC12_SI                       #
CM       = 100 * METER                         #
#---------- physical constants  (au) ----------#
H        = TWOPI                               #
HBAR     = 1.0                                 #
NA       = 6.022140857E+023                    #
C0       = C0_SI / (METER/SECOND)              #
KB       = KB_SI / (JOULE)                     #
R        = KB * NA                             #
#=====================================================================#
# Now, the following dictionaries:
#  * dpt_z2s  ==> inp: atomic number | out: atomic symbol
#  * dpt_z2m  ==> inp: atomic number | out: mass of most stable isotope
#  * dpt_z2cr ==> inp: atomic number | out: covalent radious (in bohr)
#  * dpt_s2z  ==> inp: atomic symbol | out: atomic number
#  * dpt_s2cr ==> inp: atomic symbol | out: covalent radious (in bohr)
# Atomic number (Z) to symbol
dpt_z2s = {  0:'XX' ,                                             \
             1:'H ' ,   2:'He' ,   3:'Li' ,   4:'Be' ,   5:'B ' , \
             6:'C ' ,   7:'N ' ,   8:'O ' ,   9:'F ' ,  10:'Ne' , \
            11:'Na' ,  12:'Mg' ,  13:'Al' ,  14:'Si' ,  15:'P ' , \
            16:'S ' ,  17:'Cl' ,  18:'Ar' ,  19:'K ' ,  20:'Ca' , \
            21:'Sc' ,  22:'Ti' ,  23:'V ' ,  24:'Cr' ,  25:'Mn' , \
            26:'Fe' ,  27:'Co' ,  28:'Ni' ,  29:'Cu' ,  30:'Zn' , \
            31:'Ga' ,  32:'Ge' ,  33:'As' ,  34:'Se' ,  35:'Br' , \
            36:'Kr' ,  37:'Rb' ,  38:'Sr' ,  39:'Y ' ,  40:'Zr' , \
            41:'Nb' ,  42:'Mo' ,  43:'Tc' ,  44:'Ru' ,  45:'Rh' , \
            46:'Pd' ,  47:'Ag' ,  48:'Cd' ,  49:'In' ,  50:'Sn' , \
            51:'Sb' ,  52:'Te' ,  53:'I ' ,  54:'Xe' ,  55:'Cs' , \
            56:'Ba' ,  57:'La' ,  58:'Ce' ,  59:'Pr' ,  60:'Nd' , \
            61:'Pm' ,  62:'Sm' ,  63:'Eu' ,  64:'Gd' ,  65:'Tb' , \
            66:'Dy' ,  67:'Ho' ,  68:'Er' ,  69:'Tm' ,  70:'Yb' , \
            71:'Lu' ,  72:'Hf' ,  73:'Ta' ,  74:'W ' ,  75:'Re' , \
            76:'Os' ,  77:'Ir' ,  78:'Pt' ,  79:'Au' ,  80:'Hg' , \
            81:'Tl' ,  82:'Pb' ,  83:'Bi' ,  84:'Po' ,  85:'At' , \
            86:'Rn' ,  87:'Fr' ,  88:'Ra' ,  89:'Ac' ,  90:'Th' , \
            91:'Pa' ,  92:'U ' ,  93:'Np' ,  94:'Pu' ,  95:'Am' , \
            96:'Cm' ,  97:'Bk' ,  98:'Cf' ,  99:'Es' , 100:'Fm' , \
           101:'Md' , 102:'No' , 103:'Lr'                         \
          }


# Atomic number (Z) to mass (in Da)
dpt_z2m = {  0:  0.000000 ,                                                    \
             1:  1.007825 ,   2:  4.002600 ,   3:  7.016000 ,   4:  9.012180 , \
             5: 11.009310 ,   6: 12.000000 ,   7: 14.003070 ,   8: 15.994910 , \
             9: 18.998400 ,  10: 19.992440 ,  11: 22.989800 ,  12: 23.985040 , \
            13: 26.981530 ,  14: 27.976930 ,  15: 30.973760 ,  16: 31.972070 , \
            17: 34.968850 ,  18: 39.948000 ,  19: 38.963710 ,  20: 39.962590 , \
            21: 44.955920 ,  22: 47.900000 ,  23: 50.944000 ,  24: 51.940500 , \
            25: 54.938100 ,  26: 55.934900 ,  27: 58.933200 ,  28: 57.935300 , \
            29: 62.929800 ,  30: 63.929100 ,  31: 68.925700 ,  32: 73.921900 , \
            33: 74.921600 ,  34: 79.916500 ,  35: 78.918300 ,  36: 83.911500 , \
            37: 84.911700 ,  38: 87.905600 ,  39: 89.905400 ,  40: 89.904300 , \
            41: 92.906000 ,  42: 97.905500 ,  43: 97.000000 ,  44:101.903700 , \
            45:102.904800 ,  46:105.903200 ,  47:106.904100 ,  48:113.903600 , \
            49:114.904100 ,  50:119.902200 ,  51:120.903800 ,  52:129.906700 , \
            53:126.904400 ,  54:131.904200 ,  55:132.905400 ,  56:137.905200 , \
            57:138.906300 ,  58:139.905400 ,  59:140.907600 ,  60:141.907700 , \
            61:144.912700 ,  62:151.919700 ,  63:152.921200 ,  64:157.924100 , \
            65:158.925300 ,  66:163.929200 ,  67:164.930300 ,  68:165.930300 , \
            69:168.934200 ,  70:173.938900 ,  71:174.940800 ,  72:179.946500 , \
            73:180.948000 ,  74:183.950900 ,  75:186.955700 ,  76:191.961500 , \
            77:192.962900 ,  78:194.964800 ,  79:196.966500 ,  80:201.970600 , \
            81:204.974400 ,  82:207.976600 ,  83:208.980400 ,  84:208.982400 , \
            85:209.987100 ,  86:222.017600 ,  87:223.019700 ,  88:226.025400 , \
            89:227.027800 ,  90:232.038100 ,  91:231.035900 ,  92:238.050800 , \
            93:237.048200 ,  94:244.064200 ,  95:243.061400 ,  96:247.070300 , \
            97:247.070300 ,  98:251.079600 ,  99:252.082900 , 100:257.075100 , \
           101:258.098600 , 102:259.100900 , 103:260.105300 }
#---------------------------------------------------------------------#
# Convert mass from Da to atomic units (amu)
dpt_z2m  = {k:v/AMU for k,v in dpt_z2m.items()}
#=====================================================================#
# Atomic number (Z) to covalent radious (in angstrom)
dpt_z2cr = {  0:-10.00 ,                                           \
              1:0.31 ,   2:0.28 ,   3:1.28 ,   4:0.96 ,   5:0.84 , \
              6:0.73 ,   7:0.71 ,   8:0.66 ,   9:0.57 ,  10:0.58 , \
             11:1.66 ,  12:1.41 ,  13:1.21 ,  14:1.11 ,  15:1.07 , \
             16:1.05 ,  17:1.02 ,  18:1.06 ,  19:2.03 ,  20:1.76 , \
             21:1.70 ,  22:1.60 ,  23:1.53 ,  24:1.39 ,  25:1.50 , \
             26:1.42 ,  27:1.38 ,  28:1.24 ,  29:1.32 ,  30:1.22 , \
             31:1.22 ,  32:1.20 ,  33:1.19 ,  34:1.20 ,  35:1.20 , \
             36:1.16 ,  37:2.20 ,  38:1.95 ,  39:1.90 ,  40:1.75 , \
             41:1.64 ,  42:1.54 ,  43:1.47 ,  44:1.46 ,  45:1.42 , \
             46:1.39 ,  47:1.45 ,  48:1.44 ,  49:1.42 ,  50:1.39 , \
             51:1.39 ,  52:1.38 ,  53:1.39 ,  54:1.40 ,  55:2.44 , \
             56:2.15 ,  57:2.07 ,  58:2.04 ,  59:2.03 ,  60:2.01 , \
             61:1.99 ,  62:1.98 ,  63:1.98 ,  64:1.96 ,  65:1.94 , \
             66:1.92 ,  67:1.92 ,  68:1.89 ,  69:1.90 ,  70:1.87 , \
             71:1.87 ,  72:1.75 ,  73:1.70 ,  74:1.62 ,  75:1.51 , \
             76:1.44 ,  77:1.41 ,  78:1.36 ,  79:1.36 ,  80:1.32 , \
             81:1.45 ,  82:1.46 ,  83:1.48 ,  84:1.40 ,  85:1.50 , \
             86:1.50 ,  87:2.60 ,  88:2.21 ,  89:2.15 ,  90:2.06 , \
             91:2.00 ,  92:1.96 ,  93:1.90 ,  94:1.87 ,  95:1.80 , \
             96:1.69 }
#---------------------------------------------------------------------#
# Convert radious from Angstrom to Bohr
dpt_z2cr = {k:v/ANGSTROM for k,v in dpt_z2cr.items()}
#=====================================================================#
# Create dictionaries with symbols as input
dpt_s2z  = dict( (s.strip(), z        ) for z,s in dpt_z2s.items())
dpt_s2cr = dict( (dpt_z2s[z].strip(),r) for z,r in dpt_z2cr.items() )
#=====================================================================#




#=====================================================================#
def parse_args(argv=None):

    if argv is None: argv = sys.argv[1:]

    # help requested
    if "-h" in argv or "--help" in argv:
        print(HELP)
        sys.exit(0)

    # version requested
    if "-v" in argv or "--version" in argv:
        print(PROGNAME + " " + VERSION)
        sys.exit(0)

    # Get info from command line
    fname    = [a for a in argv if not a.startswith("-")]
    ginp     = "--ginput" in argv or "-g" in argv

    # Get input file name
    if ginp: fname = None
    elif len(fname) == 0:
        sys.exit(f"ERROR: no input file given! "
                 f"Use 'python3 {PROGNAME}.py --help' for usage.")
    elif len(fname) >= 2:
        sys.exit(f"ERROR: more than one input file given! "
                 f"Use 'python3 {PROGNAME}.py --help' for usage.")
    else: fname = fname[0]

    return fname, ginp
#=====================================================================#


#===============================================#
# Some basic MATHematics                        #
#===============================================#
def distance(x1,x2):
    ''' returns distance between two points'''
    diff = [x1_i-x2_i for x1_i,x2_i in zip(x1,x2)]
    return np.linalg.norm(diff)
#-----------------------------------------------#
def angle_vecs(u,v):
    ''' returns angle between two vectors'''
    cos = np.dot(u,v) / np.linalg.norm(u) / np.linalg.norm(v)
    if abs(cos-1.0) <= EPS_SCX: cos = +1.0
    if abs(cos+1.0) <= EPS_SCX: cos = -1.0
    return np.arccos(cos)
#-----------------------------------------------#
def angle(x1,x2,x3):
    '''returns angle between three points (1-2-3)'''
    u = np.array(x1)-np.array(x2)
    v = np.array(x3)-np.array(x2)
    return angle_vecs(u,v)
#===============================================#








#===============================================#
# Diverse set of useful functions               #
#===============================================#
def flatten_llist(list_of_lists):
    ''' converts list of lists to list'''
    flattened_list = [y for x in list_of_lists for y in x]
    return flattened_list
#-----------------------------------------------#
def lowt2matrix(lowtriangle):
    if lowtriangle is None: return None
    l = len(lowtriangle)
    N = int( (-1 + (1+8*l)**0.5) / 2)
    index = 0
    matrix = [ [0.0 for ii in range(N)] for jj in range(N) ]
    for i in range(1,N+1):
        for j in range(1,i+1):
            matrix[i-1][j-1] = lowtriangle[index]
            matrix[j-1][i-1] = lowtriangle[index]
            index += 1
    return matrix
#===============================================#




#===============================================#
# Functions related to molecules                #
#===============================================#
def get_molformula(symbols):
    '''
    Returns the molecular formula of a given molecule
    '''
    formula_dict = {symbol:symbols.count(symbol) for symbol in symbols}
    molform = ""
    for key,value in sorted(formula_dict.items()):
        if value != 1: molform += "%s(%i)"%(key,value)
        if value == 1: molform += "%s"%(key)
    return molform
#---------------------------------------------#
def atonums2symbols(atonums):
    return [dpt_z2s[z].strip() for z in atonums]
#---------------------------------------------#
def atonums2masses(atonums):
    return [dpt_z2m[z] for z in atonums]
#---------------------------------------------#
def howmanyatoms(xcc): return len(xcc)//3
#---------------------------------------------#
def xyz(xcc,at): return xcc[3*at:3*at+3]
#---------------------------------------------#
def x(xcc,at)  : return xcc[3*at+0]
#---------------------------------------------#
def y(xcc,at)  : return xcc[3*at+1]
#---------------------------------------------#
def z(xcc,at)  : return xcc[3*at+2]
#---------------------------------------------#
def get_centroid(xcc,indices=None):
    if indices is None: indices = range(len(xcc)//3)
    centroid_x = sum([xcc[3*idx+0] for idx in indices])/len(indices)
    centroid_y = sum([xcc[3*idx+1] for idx in indices])/len(indices)
    centroid_z = sum([xcc[3*idx+2] for idx in indices])/len(indices)
    return [centroid_x,centroid_y,centroid_z]
#---------------------------------------------#
def get_com(xcc,masses,indices=None):
    '''
    Returns the centre of mass of the selected atoms (indices)
    If no indices given, all atoms are considered
    '''
    if indices is None: indices = range(len(masses))
    tmass = sum([           masses[idx] for idx in indices])
    com_x = sum([xcc[3*idx+0]*masses[idx] for idx in indices])/tmass
    com_y = sum([xcc[3*idx+1]*masses[idx] for idx in indices])/tmass
    com_z = sum([xcc[3*idx+2]*masses[idx] for idx in indices])/tmass
    return [com_x,com_y,com_z]
#---------------------------------------------#
def get_distmatrix(xcc):
    nat = howmanyatoms(xcc)
    dmatrix = np.zeros( (nat,nat) )
    for ii in range(nat):
        xii = xyz(xcc,ii)
        for jj in range(ii+1,nat):
            xjj = xyz(xcc,jj)
            dmatrix[ii][jj] = distance(xii,xjj)
            dmatrix[jj][ii] = dmatrix[ii][jj]
    return dmatrix
#---------------------------------------------#
def set_origin(xcc,x0):
    '''
    returns xcc with x0 as origin
    '''
    nat = howmanyatoms(xcc)
    return [xi-xj for xi,xj in zip(xcc,nat*x0)]
#---------------------------------------------#
def shift2com(xcc,masses):
    '''
    Function to shift to center of mass
    '''
    com = get_com(xcc,masses)
    xcc = set_origin(xcc,com)
    return xcc
#---------------------------------------------#
def islinear(xcc):
    nats    = howmanyatoms(xcc)
    masses  = [1.0 for at in range(nats)]
    itensor = get_itensor_matrix(xcc,masses)
    linear  = get_itensor_evals(itensor)[3]
    return linear
#---------------------------------------------#
def center_and_orient(xcc,gcc,Fcc,masses):
    # number of atoms?
    nats = howmanyatoms(xcc)
    # in center of mass
    xcc = shift2com(xcc,masses)
    if nats == 1: return xcc, gcc, Fcc
    # Is it linear?
    itensor = get_itensor_matrix(xcc,masses)
    evalsI, rotTs, rtype, linear = get_itensor_evals(itensor)
    if not linear: return xcc, gcc, Fcc
    # Get moments and axis of inertia
    itensor = np.matrix(itensor)
    evalsI, evecsI = np.linalg.eigh(itensor)
    evecsI = np.matrix(evecsI)
    # Rotation matrix 3N x 3N
    zeros = np.zeros((3, 3))
    R= []
    for at in range(nats):
        row = []
        count = 0
        while count < at:
              row.append(zeros)
              count += 1
        row.append(evecsI)
        while count < nats-1:
              row.append(zeros)
              count += 1
        R.append(row)
    R = np.matrix(np.block(R))
    # Rotate coordinates
    xcc = np.matrix(xcc) * R
    xcc = xcc.tolist()[0]
    # Rotate gradient
    if gcc is not None and len(gcc) != 0:
       gcc = np.matrix(gcc) * R
       gcc = gcc.tolist()[0]
    # rotate Fcc
    if Fcc is not None and len(Fcc) != 0:
       nr,nc = np.matrix(Fcc).shape
       if nr != nc: Fcc = lowt2matrix(Fcc)
       Fcc = R.transpose() * np.matrix(Fcc) * R
       Fcc = Fcc.tolist()
    return xcc, gcc, Fcc
#---------------------------------------------#
def cc2ms_x(xcc,masses,mu=1.0/AMU):
    ''' cartesian --> mass-scaled; x'''
    if xcc is None or len(xcc) == 0: return xcc
    nat  = len(masses)
    cfs  = [ (masses[idx]/mu)**0.5 for idx in range(nat)]
    xms  = [ [x(xcc,at)*cfs[at],y(xcc,at)*cfs[at],z(xcc,at)*cfs[at]] for at in range(nat)]
    xms  = flatten_llist(xms)
    return xms
#---------------------------------------------#
def cc2ms_g(gcc,masses,mu=1.0/AMU):
    ''' cartesian --> mass-scaled; gradient'''
    if gcc is None or len(gcc) == 0: return gcc
    nat  = len(masses)
    cfs  = [ (masses[idx]/mu)**0.5 for idx in range(nat)]
    gms  = [ [x(gcc,at)/cfs[at],y(gcc,at)/cfs[at],z(gcc,at)/cfs[at]] for at in range(nat)]
    gms  = flatten_llist(gms)
    return gms
#---------------------------------------------#
def cc2ms_F(Fcc,masses,mu=1.0/AMU):
    ''' cartesian --> mass-scaled; force constant matrix'''
    if Fcc is None or len(Fcc) == 0: return Fcc
    nat    = len(masses)
    Fms    = [ [ 0.0 for at1 in range(3*nat) ] for at2 in range(3*nat)]
    for i in range(3*nat):
        mi = masses[int(i/3)]
        for j in range(3*nat):
            mj = masses[int(j/3)]
            f = mu / ((mi*mj)**0.5)
            Fms[i][j] = Fcc[i][j] * f
    return Fms
#===============================================#




#===============================================#
# Functions related to frequencies              #
#===============================================#
def afreq2wnum(angfreq):
    return angfreq / TWOPI / C0
#-----------------------------------------------#
def eval2afreq(evalue,mu=1.0/AMU):
    return np.sign(evalue) * (abs(evalue)/mu)**0.5
#-----------------------------------------------#
def afreq2cm(angfreq):
    return afreq2wnum(angfreq)/CM
#-----------------------------------------------#
def same_freqs(ccfreqs,icfreqs):
    # compare lengths
    if len(ccfreqs) != len(icfreqs):
        return False
    # compare freqs
    for ccf, icf in zip(ccfreqs,icfreqs):
        ccf = afreq2cm(ccf)
        icf = afreq2cm(icf)
        if abs(ccf-icf) > EPS_CCIC: return False
    return True
#===============================================#




#===============================================#
# Rotations/Vibrations                          #
#===============================================#
def get_itensor_matrix(xcc,masses):
    ''' returns inertia tensor (au)'''
    nat = howmanyatoms(xcc)
    inertia = [[0.0 for i in range(3)] for j in range(3)]
    for i in range(nat):
        # Diagonal elements
        inertia[0][0] += masses[i] * (y(xcc,i)**2 + z(xcc,i)**2)
        inertia[1][1] += masses[i] * (z(xcc,i)**2 + x(xcc,i)**2)
        inertia[2][2] += masses[i] * (x(xcc,i)**2 + y(xcc,i)**2)
        # Offdiagonal elements
        inertia[0][1] -= masses[i] * x(xcc,i) * y(xcc,i)
        inertia[0][2] -= masses[i] * z(xcc,i) * x(xcc,i)
        inertia[1][2] -= masses[i] * y(xcc,i) * z(xcc,i)
    inertia[1][0] = inertia[0][1]
    inertia[2][0] = inertia[0][2]
    inertia[2][1] = inertia[1][2]
    return inertia
#---------------------------------------------#
def get_itensor_evals(itensor):
    evalsI, evecsI = np.linalg.eigh(itensor)

    Ia, Ib, Ic = evalsI

    bool_a  = abs(Ia)        < EPS_INERTIA # Ia = 0
    bool_ab = abs(Ia/Ib-1.0) < EPS_FLOAT   # Ia = Ib
    bool_bc = abs(Ib/Ic-1.0) < EPS_FLOAT   # Ib = Ic
    bool_abc = bool_ab and bool_bc

    if   bool_abc  : rtype = "spherical top"
    elif bool_ab   : rtype = "oblate symmetric top"
    elif bool_bc   :
         if  bool_a: rtype = "linear rotor"
         else      : rtype = "prolate symmetric top"
    else           : rtype = "asymmetric top"

    if rtype == "linear rotor":
       linear = True
       evalsI = [evalsI[1]]
    else:
       linear = False

    # rotational temperature
    rotTs = [HBAR**2 / (2*Ii*KB) for Ii in evalsI]
    return evalsI, rotTs, rtype, linear
#-----------------------------------------------#
def get_projectionmatrix(xcc,masses,v0=None):
    '''
    Generates matrix to project translation
    and rotation coordinates (mass scaled/weighted)
    Other coordinate can be projected by introducing it
    using v0 (in mass-scaled)
    '''
    nat  = len(masses)
    # translation
    sqrtmasses = [np.sqrt(mass) for mass in masses]
    b1 = [term if ii==0 else 0.0 for term in sqrtmasses for ii in range(3)]
    b2 = [term if ii==1 else 0.0 for term in sqrtmasses for ii in range(3)]
    b3 = [term if ii==2 else 0.0 for term in sqrtmasses for ii in range(3)]
    norm1 = np.linalg.norm(b1)
    norm2 = np.linalg.norm(b2)
    norm3 = np.linalg.norm(b3)
    b1 /= norm1
    b2 /= norm2
    b3 /= norm3
    vecs = [b1,b2,b3]
    # rotation
    b4 = np.zeros(len(xcc))
    b5 = np.zeros(len(xcc))
    b6 = np.zeros(len(xcc))
    for i in range(nat):
        b4[3*i + 1] =   np.sqrt(masses[i]) * z(xcc,i)
        b4[3*i + 2] = - np.sqrt(masses[i]) * y(xcc,i)
        b5[3*i + 0] = - np.sqrt(masses[i]) * z(xcc,i)
        b5[3*i + 2] =   np.sqrt(masses[i]) * x(xcc,i)
        b6[3*i + 0] =   np.sqrt(masses[i]) * y(xcc,i)
        b6[3*i + 1] = - np.sqrt(masses[i]) * x(xcc,i)
    norm4 = np.linalg.norm(b4)
    norm5 = np.linalg.norm(b5)
    norm6 = np.linalg.norm(b6)
    if norm4 > EPS_NORM: b4 /= norm4; vecs.append(b4)
    if norm5 > EPS_NORM: b5 /= norm5; vecs.append(b5)
    if norm6 > EPS_NORM: b6 /= norm6; vecs.append(b6)
    # Gram Schmidt
    X = np.matrix(vecs).transpose()
    X_gs, R = np.linalg.qr(X)
    projmatrix = X_gs * X_gs.H
    if v0 is not None:
       normv0 = np.linalg.norm(v0)
       if normv0 > EPS_NORM:
          v0 = np.matrix( v0 ) / normv0
          projmatrix += v0.transpose() * v0
    return projmatrix
#-----------------------------------------------#
def project_hessian(Fms,natoms,proj_matrix):
    ''' Fms: hessian in mass-scaled '''
    # identity matrix
    I = np.identity(3*natoms)
    # Calculate projected hessian matrix
    Fms_proj = (I - proj_matrix) * Fms * (I - proj_matrix)
    return Fms_proj
#-----------------------------------------------#
def diagonalize_hessian(Fms,mu=1.0/AMU):
    # as Fms is symmetric --> diagonalize with eigh
    evalsF, evecsF = np.linalg.eigh(Fms)
    # Convert evals to angular frequencies
    ccfreqs = [eval2afreq(v,mu) for v  in evalsF]
    # evecsF to list of eigenvectors
    evecsF  = evecsF.transpose().tolist()
    # return data
    return ccfreqs, evalsF, evecsF
#-----------------------------------------------#
def detect_frozen(Fcc,nat):
    '''
    Fcc --> 3Nx3N
    '''
    frozen =  []
    if Fcc is None or len(Fcc) == 0: return frozen
    for at in range(nat):
        # get columns (rows are equivalent; symmetric matrix)
        colx = Fcc[:][3*at+0][0]
        coly = Fcc[:][3*at+1][0]
        colz = Fcc[:][3*at+2][0]
        # Get norm of column
        normx = np.linalg.norm(colx)
        normy = np.linalg.norm(coly)
        normz = np.linalg.norm(colz)
        # are columns made of zeros?
        if normx != 0.0: continue
        if normy != 0.0: continue
        if normz != 0.0: continue
        frozen.append(at)
    return frozen
#-----------------------------------------------#
def calc_ccfreqs(Fcc,masses,xcc,mu=1.0/AMU,v0=None):
    '''
    xcc has to be centered at com
    v0 in case gradient has to be removed
    '''
    # num of atoms and Fcc in matrix format
    nat = len(masses)
    if len(Fcc) != 3*nat: Fcc = lowt2matrix(Fcc)
    # Frozen atoms?
    frozen  = detect_frozen(Fcc,nat)
    boolN   = np.array([at not in frozen for at in range(nat)])
    bool3N  = np.array([at not in frozen for at in range(nat) for ii in range(3)])
    # if FROZEN atoms, then reduce system!!
    if len(frozen) != 0:
       masses = np.array(masses)[boolN]
       xcc    = np.array(xcc)[bool3N]
       if v0 is not None: v0 = np.array(v0)[bool3N]
       # force constant matrix
       Fcc = [[Fcc[idx1][idx2] for idx1 in range(3*nat) if bool3N[idx1]]\
                               for idx2 in range(3*nat) if bool3N[idx2]]
       # num atoms
       nat    = len(masses)
       # re-center system
       xcc = shift2com(xcc,masses)
    # Analyze system
    linear = islinear(xcc)
    if linear: nvdof = 3*nat-5
    else     : nvdof = 3*nat-6
    if v0 is not None: nvdof -= 1
    # mass-scaled hessian
    Fms = cc2ms_F(Fcc,masses,mu=mu)
    Fms = np.matrix(Fms)
    # projection matrix
    pmatrix = get_projectionmatrix(xcc,masses,v0)
    # projected hessian
    Fms = project_hessian(Fms,nat,pmatrix)
    # Diagonalization
    ccfreqs, evalsF, evecsF = diagonalize_hessian(Fms,mu)
    # remove extra frequencies
    idxs = sorted([(abs(fq),fq,idx) for idx,fq in enumerate(ccfreqs)])
    idxs.reverse()
    idxs = idxs[:nvdof]
    idxs = sorted([(fq,idx) for absfq,fq,idx in idxs])
    idxs = [idx for fq,idx in idxs]
    ccfreqs = [ccfreqs[idx] for idx in idxs]
    evalsF  = [ evalsF[idx] for idx in idxs]
    evecsF  = [ evecsF[idx] for idx in idxs]
    # Consider the removed atoms in the eigenvectors
    if len(frozen) != 0:
       for idx,evecF in enumerate(evecsF):
           evecsF[idx] = [evecF.pop(0) if booli else 0.0 for booli in bool3N]
    return ccfreqs, evalsF, evecsF
#-----------------------------------------------#
def scale_freqs(freqs,fscal):
    return [freq*fscal for freq in freqs]
#===============================================#






#===============================================#
# Internal coordinates / Graph Theory           #
#===============================================#
def ic2string(ic):
    ictype, icatoms = ic
    if ictype == "st" : return "-".join(["%i"%(at+1) for at in icatoms])
    if ictype == "ab" : return "-".join(["%i"%(at+1) for at in icatoms])
    if ictype == "pt" : return "-".join(["%i"%(at+1) for at in icatoms])
    if ictype == "lb" : return "=".join(["%i"%(at+1) for at in icatoms])
    if ictype == "it" : return "_".join(["%i"%(at+1) for at in icatoms])
    if ictype == "rax": return "rax("+"-".join(["%i"%(at+1) for at in icatoms])+")"
    if ictype == "lb_a": return "=".join(["%i"%(at+1) for at in icatoms])+"(a)"
    if ictype == "lb_b": return "=".join(["%i"%(at+1) for at in icatoms])+"(b)"
    return ictype
#---------------------------------------------#
def merge_ics(ics_st,ics_ab,ics_lb,ics_it,ics_pt):
    all_ics  = [ ("st",ic) for ic in sorted(ics_st)] # stretching
    all_ics += [ ("ab",ic) for ic in sorted(ics_ab)] # angular bending
    all_ics += [ ("lb",ic) for ic in sorted(ics_lb)] # linear  bending
    all_ics += [ ("it",ic) for ic in sorted(ics_it)] # improper torsion
    all_ics += [ ("pt",ic) for ic in        ics_pt ] # proper   torsion (DO NOT SORT)
    return all_ics
#-----------------------------------------------#
def unmerge_ics(all_ics):
    ics_st = [ ic for ic_type,ic in all_ics if ic_type=="st"]
    ics_ab = [ ic for ic_type,ic in all_ics if ic_type=="ab"]
    ics_lb = [ ic for ic_type,ic in all_ics if ic_type=="lb"]
    ics_it = [ ic for ic_type,ic in all_ics if ic_type=="it"]
    ics_pt = [ ic for ic_type,ic in all_ics if ic_type=="pt"]
    return ics_st,ics_ab,ics_lb,ics_it,ics_pt
#-----------------------------------------------#
def count_ics(all_ics):
    ics_st,ics_ab,ics_lb,ics_it,ics_pt = unmerge_ics(all_ics)
    nICs = len(ics_st)+len(ics_ab)+2*len(ics_lb)+len(ics_it)+len(ics_pt)
    return nICs
#-----------------------------------------------#
def get_adjmatrix(xcc,symbols,scale=CONNECTSCAL,mode="bool"):
    '''
    returns adjacency matrix (connection matrix);
    also distance matrix and number of bonds
    * mode = bool, int
    '''
    nbonds  = 0
    nat     = howmanyatoms(xcc)
    dmatrix = get_distmatrix(xcc)
    if   mode == "int" : no, yes = 0    , 1
    elif mode == "bool": no, yes = False, True
    else               : no, yes = False, True
    cmatrix = [ [no for ii in range(nat)] for jj in range(nat)]
    for ii in range(nat):
        cr_ii = dpt_s2cr[symbols[ii]] # covalent radius
        for jj in range(ii+1,nat):
            cr_jj = dpt_s2cr[symbols[jj]]
            dref  = (cr_ii+cr_jj)*scale
            if dmatrix[ii][jj] < dref:
               nbonds += 1
               cmatrix[ii][jj] = yes
               cmatrix[jj][ii] = yes
    return cmatrix, dmatrix, nbonds
#-----------------------------------------------#
def get_numbonds(amatrix):
    nbonds = 0
    nnodes = len(amatrix)
    for node1 in range(nnodes):
        for node2 in range(node1+1,nnodes):
            if amatrix[node1][node2] in [True,1]: nbonds += 1
    return nbonds
#-----------------------------------------------#
def adjacency_matrix2list(amatrix):
    alist = {}
    for node1,row in enumerate(amatrix):
       alist[node1]= [node2 for node2, bonded in enumerate(row) if bonded in [True,1]]
    return alist
#-----------------------------------------------#
def get_subgraph(node,alist,fragment=[]):
    fragment += [node]
    neighbors = alist[node]
    for neighbor in neighbors:
        if neighbor not in fragment:
            fragment = get_subgraph(neighbor,alist,fragment)
    return fragment
#-----------------------------------------------#
def get_fragments(alist):
    fragments = []
    visited   = set([])
    for node in alist.keys():
        if node in visited: continue
        fragment = set(get_subgraph(node,alist,[]))
        if fragment not in fragments: fragments.append(fragment)
        visited = visited.union(fragment)
    return fragments
#-----------------------------------------------#
def distance_2fragments(frg1,frg2,xcc):
    min_dist = float("inf")
    pair     = (None,None)
    for at1 in frg1:
        x1 = xyz(xcc,at1)
        for at2 in frg2:
            x2 = xyz(xcc,at2)
            dist = distance(x1,x2)
            if dist < min_dist:
               min_dist = dist
               pair     = (at1,at2)
    return min_dist, pair
#-----------------------------------------------#
def distance_allfragments(fragments,xcc):
    nfrags = len(fragments)
    the_list = []
    for idx1 in range(nfrags):
        frg1 = fragments[idx1]
        for idx2 in range(idx1+1,nfrags):
            frg2 = fragments[idx2]
            dist, (at1,at2) = distance_2fragments(frg1,frg2,xcc)
            the_list.append( (dist,idx1,idx2,at1,at2) )
    return sorted(the_list)
#-----------------------------------------------#
def frags_distances(fragments):
    ''' use distance_allfragments -  this one returns distance
        of 1 and just first atom in each fragment'''
    fdists = []
    nfrags = len(fragments)
    for idx1 in range(nfrags):
        for idx2 in range(idx1+1,nfrags):
            dist = 1.0
            atf1 = list(fragments[idx1])[0]
            atf2 = list(fragments[idx2])[0]
            fdists.append( (1.0,idx1,idx2,atf1,atf2) )
    fdists.sort()
    return fdists
#-----------------------------------------------#
def link_fragments(xcc,amatrix,nfrags=1):
    if   amatrix[0][0] is False: bonded = True
    elif amatrix[0][0] == 0    : bonded = 1
    else: exit("sth wrong in adjacency matrix!")
    alist     = adjacency_matrix2list(amatrix)
    fragments = get_fragments(alist)
   #fdists    = frags_distances(fragments)
    fdists    = distance_allfragments(fragments,xcc)
    inumfrags = len(fragments)
    fnumfrags = len(fragments)
    for dist,idx1,idx2,atf1,atf2 in fdists:
        fragments[idx1] = fragments[idx1].union(fragments[idx2])
        fragments[idx2] = set([])
        amatrix[atf1][atf2] = bonded
        amatrix[atf2][atf1] = bonded
        fnumfrags = sum([1 for frag in fragments if len(frag)!=0])
        if fnumfrags == nfrags: break
    fragments = [frag for frag in fragments if len(frag) != 0]
    return amatrix, fragments, inumfrags, fnumfrags
#-----------------------------------------------#
def ics_get_stretchings(cmatrix,natoms):
   ics_st = [(at1,at2) for at1 in range(natoms) for at2 in range(at1,natoms) if cmatrix[at1][at2] in [True,1]]
   return ics_st
#-----------------------------------------------#
def ics_get_iccentral(adj_list):
    ic_3ats = []
    ic_4ats = []
    for at2 in adj_list.keys():
        bonded  = adj_list[at2]
        nbonded = len(bonded)
        if nbonded < 2: continue
        for idx1 in range(nbonded):
            for idx3 in range(idx1+1,nbonded):
                bending = (bonded[idx1],at2,bonded[idx3])
                ic_3ats.append(bending)
                if nbonded < 3: continue
                for idx4 in range(idx3+1,nbonded):
                    improper_torsion = (bonded[idx1],bonded[idx3],bonded[idx4],at2)
                    ic_4ats.append(improper_torsion)
    return ic_3ats, ic_4ats
#-----------------------------------------------#
def ics_classify_bends(xcc,ic_3ats):
    ics_lb = []
    ics_ab = []
    thetas     = {}
    for at1,at2,at3 in ic_3ats:
       x1 = xyz(xcc,at1)
       x2 = xyz(xcc,at2)
       x3 = xyz(xcc,at3)
       theta = abs(np.rad2deg(angle(x1,x2,x3)))
       if theta < EPS_LINEAR or theta > 180-EPS_LINEAR:
          ics_lb.append( (at1,at2,at3) )
       else:
          ics_ab.append( (at1,at2,at3) )
       thetas[(at1,at2,at3)] = theta
    return ics_lb, ics_ab, thetas
#-----------------------------------------------#
def ics_get_ptorsions(ics_st,adj_list,xcc):
    ics_pt = []
    for at2,at3 in ics_st:
        x2 = xyz(xcc,at2)
        x3 = xyz(xcc,at3)
        bondedto2 = list(adj_list[at2])
        bondedto3 = list(adj_list[at3])
        bondedto2.remove(at3)
        bondedto3.remove(at2)
        if len(bondedto2) == 0: continue
        if len(bondedto3) == 0: continue
        for at1 in bondedto2:
            x1 = xyz(xcc,at1)
            for at4 in bondedto3:
                if at1 == at4: continue
                x4 = xyz(xcc,at4)
                # the two angles
                angA  = np.rad2deg(angle(x1,x2,x3))
                angB  = np.rad2deg(angle(x2,x3,x4))
                # linear?
                booleanA = angA < EPS_LINEAR or angA > 180-EPS_LINEAR
                booleanB = angB < EPS_LINEAR or angB > 180-EPS_LINEAR
                if booleanA or booleanB: continue
                ptorsion = (at1,at2,at3,at4)
                ics_pt.append(ptorsion)
    return ics_pt
#-----------------------------------------------#
def ics_get_ltorsions(ics_lb,adj_list):
    ic_ltors = []
    for at1,at2,at3 in ics_lb:
        bondedto1 = list(adj_list[at1])
        bondedto3 = list(adj_list[at3])
        if at1 in bondedto3: bondedto3.remove(at1)
        if at2 in bondedto3: bondedto3.remove(at2)
        if at2 in bondedto1: bondedto1.remove(at2)
        if at3 in bondedto1: bondedto1.remove(at3)
        for at0 in bondedto1:
            for at4 in bondedto3:
                if at0 == at4: continue
                ltorsion = (at0,at1,at3,at4)
                ic_ltors.append(ltorsion)
    return ic_ltors
#-----------------------------------------------#
def ics_from_geom(xcc,symbols,scale=CONNECTSCAL,nfrags=1,verbose=False):
    natoms = len(symbols)
    amatrix, dmatrix, nbonds = get_adjmatrix(xcc,symbols,scale=scale,mode="bool")
    amatrix, fragments, inumfrags, fnumfrags = link_fragments(xcc,amatrix,nfrags=nfrags)
    if inumfrags != fnumfrags: nbonds = get_numbonds(amatrix)
    alist = adjacency_matrix2list(amatrix)
    ics_st = ics_get_stretchings(amatrix,natoms)
    ic_3ats, ics_it = ics_get_iccentral(alist)
    ics_lb, ics_ab, angles = ics_classify_bends(xcc,ic_3ats)
    ics_pt  = ics_get_ptorsions(ics_st,alist,xcc)
    ics_pt += ics_get_ltorsions(ics_lb,alist)
    # return data
    ics = merge_ics(ics_st,ics_ab,ics_lb,ics_it,ics_pt)
    if not verbose: return ics
    else: return ics, amatrix
#-----------------------------------------------#
def ics_depure_bendings(ic_3ats,keep=[]):
    ''' up to 3 bendings per central atom'''
    centers = [at2 for at1,at2,at3 in keep]
    random.shuffle(ic_3ats)
    for at1,at2,at3 in ic_3ats:
        if centers.count(at2) >= 3: continue
        #if at2 in centers: continue
        keep = [(at1,at2,at3)]+keep
        centers.append(at2)
    return keep
#-----------------------------------------------#
def ics_depure_itorsions(ic_4ats,keep=[]):
    ''' up to two improper torsions per atom'''
    centers = [at4 for at1,at2,at3,at4 in keep]
    random.shuffle(ic_4ats)
    for at1,at2,at3,at4 in ic_4ats:
        if centers.count(at4) >= 2: continue
        #if at4 in centers: continue
        keep = [(at1,at2,at3,at4)]+keep
        centers.append(at4)
    return keep
#-----------------------------------------------#
def ics_depure_ptorsions(ic_4ats,keep=[]):
    ''' one torsion per bond'''
    centers = [(at2,at3) for at1,at2,at3,at4 in keep]
    random.shuffle(ic_4ats)
    for at1,at2,at3,at4 in ic_4ats:
        if (at2,at3) in centers: continue
        if (at3,at2) in centers: continue
        keep = [(at1,at2,at3,at4)]+keep
        centers.append((at2,at3))
    return keep
#-----------------------------------------------#
def ics_depure(ics,keep=[]):
    ''' keep: those that cannot be removed '''
    ics_st ,ics_ab ,ics_lb ,ics_it ,ics_pt  = unmerge_ics(ics)
    keep_st,keep_ab,keep_lb,keep_it,keep_pt = unmerge_ics(keep)
    # depure angular bendings
    ics_ab = ics_depure_bendings(ics_ab,keep_ab)
    # depure torsions
    ics_it = ics_depure_itorsions(ics_it,keep_it)
    ics_pt = ics_depure_ptorsions(ics_pt,keep_pt)
    # merge again
    ics = merge_ics(ics_st,ics_ab,ics_lb,ics_it,ics_pt)
    return ics
#-----------------------------------------------#
#===============================================#


#===============================================#
# WILSON method for freqs in internal coords    #
#===============================================#
def wilson_bvecs(xcc):
    '''
    This functions calculates the distance
    between each pair of atoms (dij) and also
    the unit bond vector eij = (rj - ri)/dij
    '''
    nat = howmanyatoms(xcc)
    bond_vectors = {}
    for ii in range(nat):
        ri = np.array(xyz(xcc,ii))
        for jj in range(ii+1,nat):
            rj = np.array(xyz(xcc,jj))
            eij = rj-ri
            dij = np.linalg.norm(eij)
            eij = eij / dij
            bond_vectors[(ii,jj)] = ( eij, dij)
            bond_vectors[(jj,ii)] = (-eij, dij)
    return bond_vectors
#-----------------------------------------------#
def wilson_stretch(bond_vectors,ij,natoms):
    '''
    Returns the row of the B matrix associated to
    the bond length between atom i and atom j and
    also the corresponding C matrix.
    Check ref [1] and [5].
    '''
    # Using nomenclature of reference [5]
    n = min(ij)
    m = max(ij)
    u, r = bond_vectors[(n,m)]
    # Calculating 1st derivatives: row of B matrix
    B_row = [0.0 for idx in range(3*natoms)]
    for a in [m,n]:
        if a == m: zeta_amn = +1.0
        if a == n: zeta_amn = -1.0
        for i in [0,1,2]:
            dr_dai = zeta_amn * u[i]
            B_row[3*a+i] = dr_dai
    # Calculating 2nd derivatives: 2D matrix of C tensor
    C_matrix = [ [0.0 for idx1 in range(3*natoms)] for idx2 in range(3*natoms)]
    #C_matrix = np.zeros( (3*natoms,3*natoms) )
    for a in [m,n]:
      for i in [0,1,2]:
          for b in [m,n]:
            for j in [0,1,2]:
               if C_matrix[3*a+i][3*b+j] != 0.0: continue
               # Get delta values
               if a == b: delta_ab = 1.0
               else:      delta_ab = 0.0
               if i == j: delta_ij = 1.0
               else:      delta_ij = 0.0
               # Get C element
               dr_daidbj = ((-1.0) ** delta_ab) * (u[i]*u[j] - delta_ij) / r
               # Save data in both positions
               C_matrix[3*a+i][3*b+j] = dr_daidbj
               C_matrix[3*b+j][3*a+i] = dr_daidbj
    return [B_row], [C_matrix]
#-----------------------------------------------#
def wilson_abend(bond_vectors,ijk,natoms):
    '''
    Returns the row of the B matrix associated to the
    i-j-k angle bend and the corresponding C matrix.
    Check ref [1] and [5].
    '''
    # Using nomenclature of reference [5]
    m, o, n = ijk
    u, lambda_u = bond_vectors[(o,m)]
    v, lambda_v = bond_vectors[(o,n)]
    # Get internal coordinate: bond angle
    q = angle_vecs(u,v)
    sinq = np.sin(q)
    cosq = np.cos(q)
    # Generation of w
    w = np.cross(u,v)
    w = w / np.linalg.norm(w)
    uxw = np.cross(u,w)
    wxv = np.cross(w,v)
    # Calculating 1st derivatives: row of B matrix
    B_row = [0.0 for idx in range(3*natoms)]
    for a in [m,o,n]:
        # Get zeta values
        if a == m: zeta_amo = +1.0; zeta_ano =  0.0
        if a == o: zeta_amo = -1.0; zeta_ano = -1.0
        if a == n: zeta_amo =  0.0; zeta_ano = +1.0
        for i in [0,1,2]:
            # Get B element
            dq_dai =  zeta_amo * uxw[i] / lambda_u + zeta_ano * wxv[i] / lambda_v
            B_row[3*a+i] = dq_dai
    # Calculating 2nd derivatives: 2D matrix of C tensor
    #C_matrix = np.zeros( (3*natoms,3*natoms) )
    C_matrix = [ [0.0 for idx1 in range(3*natoms)] for idx2 in range(3*natoms)]
    if abs(sinq) < EPS_SCX: return [B_row], [C_matrix]
    for a in [m,o,n]:
      for i in [0,1,2]:
        for b in [m,o,n]:
          for j in [0,1,2]:
            if C_matrix[3*a+i][3*b+j] != 0.0: continue
            # Define all delta and zeta values
            if a == m: zeta_amo = +1.0; zeta_ano =  0.0
            if a == o: zeta_amo = -1.0; zeta_ano = -1.0
            if a == n: zeta_amo =  0.0; zeta_ano = +1.0
            if b == m: zeta_bmo = +1.0; zeta_bno =  0.0
            if b == o: zeta_bmo = -1.0; zeta_bno = -1.0
            if b == n: zeta_bmo =  0.0; zeta_bno = +1.0
            if i == j: delta_ij = 1.0
            else:      delta_ij = 0.0
            # Get second derivative
            t1 = zeta_amo*zeta_bmo*(u[i]*v[j]+u[j]*v[i]-3*u[i]*u[j]*cosq+delta_ij*cosq)/(lambda_u**2 * sinq)
            t2 = zeta_ano*zeta_bno*(v[i]*u[j]+v[j]*u[i]-3*v[i]*v[j]*cosq+delta_ij*cosq)/(lambda_v**2 * sinq)
            t3 = zeta_amo*zeta_bno*(u[i]*u[j]+v[j]*v[i]-u[i]*v[j]*cosq-delta_ij)/(lambda_u*lambda_v*sinq)
            t4 = zeta_ano*zeta_bmo*(v[i]*v[j]+u[j]*u[i]-v[i]*u[j]*cosq-delta_ij)/(lambda_u*lambda_v*sinq)
            t5 = cosq / sinq * B_row[3*a+i] * B_row[3*b+j]
            dr_daidbj = t1 + t2 + t3 + t4 - t5
            C_matrix[3*a+i][3*b+j] = dr_daidbj
            C_matrix[3*b+j][3*a+i] = dr_daidbj
    return [B_row], [C_matrix]
#-----------------------------------------------#
def wilson_auxlinB(m,o,n,k):
        om = m - o
        on = n - o
        dom = np.linalg.norm(om)
        don = np.linalg.norm(on)
        qk = ((n[2]-o[2])*(m[k]-o[k]) - (n[k]-o[k])*(m[2]-o[2])) / dom / don
        Bk = []
        Dk = []
        for a in ["m","o","n"]:
            for i in [0,1,2]:
                if a == "m": dam, dao, dan = 1.0, 0.0, 0.0
                if a == "o": dam, dao, dan = 0.0, 1.0, 0.0
                if a == "n": dam, dao, dan = 0.0, 0.0, 1.0
                if i ==  2 : di2 = 1.0
                else       : di2 = 0.0
                if i == k  : dik = 1.0
                else       : dik = 0.0
                # Numerator
                N_ai = dik*(dao-dan)*m[2] + di2*(dan-dao)*m[k] +\
                       dik*(dan-dam)*o[2] + di2*(dam-dan)*o[k] +\
                       dik*(dam-dao)*n[2] + di2*(dao-dam)*n[k]
                # Denominator
                D_ai = (dam-dao)*(m[i]-o[i]) * don / dom + \
                       (dan-dao)*(n[i]-o[i]) * dom / don
                Dk.append(D_ai)
                # Whole derivative
                B_ai = (N_ai - qk*D_ai) / (dom*don)
                Bk.append(B_ai)
        return np.array(Bk), np.array(Dk)
#-----------------------------------------------#
def wilson_auxlinC(m,o,n,k,Bk=None,Dk=None):
        om = m - o
        on = n - o
        dom = np.linalg.norm(om)
        don = np.linalg.norm(on)
        if (Bk is None) or (Dk is None): Bk, Dk = get_B(m,o,n,k,True)
        Ck = np.zeros( (9,9) )
        for a in ["m","o","n"]:
            for i in [0,1,2]:
                if a == "m": dam, dao, dan = 1.0, 0.0, 0.0; A=0
                if a == "o": dam, dao, dan = 0.0, 1.0, 0.0; A=1
                if a == "n": dam, dao, dan = 0.0, 0.0, 1.0; A=2
                if i ==  2 : di2 = 1.0
                else       : di2 = 0.0
                if i == k  : dik = 1.0
                else       : dik = 0.0
                for b in ["m","o","n"]:
                    for j in [0,1,2]:
                        if b == "m": dbm, dbo, dbn = 1.0, 0.0, 0.0; B=0
                        if b == "o": dbm, dbo, dbn = 0.0, 1.0, 0.0; B=1
                        if b == "n": dbm, dbo, dbn = 0.0, 0.0, 1.0; B=2
                        if j ==  2 : dj2 = 1.0
                        else       : dj2 = 0.0
                        if j == k  : djk = 1.0
                        else       : djk = 0.0
                        # Term 1
                        term1 = (djk*di2-dik*dj2)*( (dan-dao)*dbm + (dam-dan)*dbo + (dao-dam)*dbn )
                        # Term 2
                        term2 = Bk[3*B+j] * Dk[3*A+i]
                        # Term 3
                        term3 = Bk[3*A+i] * Dk[3*B+j]
                        #
                        Ck[3*A+i,3*B+j] = (term1 - term2 - term3) / dom / don
        return np.matrix(Ck)
#-----------------------------------------------#
def wilson_lbend(m,o,n,MON,natoms):
    M,O,N = MON
    #------------------------------------------#
    # Redefine new Cartesian coordinate system #
    #------------------------------------------#
    # Define new z axis
    new_z = n - m
    new_z = new_z / np.linalg.norm(new_z)

    # Define new y axis
    ref  = np.array([+1.0,+1.0,+1.0])
    prod = np.dot(new_z,ref) / np.linalg.norm(new_z) / np.linalg.norm(ref)
    if abs(prod) == 0.0:
       ref = np.array([+1.0,-1.0,0.0])
    new_y = np.cross(ref,new_z)
    new_y = new_y / np.linalg.norm(new_y)

    # Define new x axis
    new_x = np.cross(new_y,new_z)
    new_x = new_x / np.linalg.norm(new_x)

    # Define rotation matrix such as new_r = R * old_r,
    # considering new_r and old_r as column vectors
    R = np.matrix([new_x,new_y,new_z])
    Tbar = np.zeros( (9,9)  )
    Tbar[0:3,0:3] = R
    Tbar[3:6,3:6] = R
    Tbar[6:9,6:9] = R

    # Define position vectors with this new matrix
    m = R * np.matrix(m).transpose(); m = np.array( m.transpose().tolist()[0] )
    o = R * np.matrix(o).transpose(); o = np.array( o.transpose().tolist()[0] )
    n = R * np.matrix(n).transpose(); n = np.array( n.transpose().tolist()[0] )

    #-----------------------------------------------#
    # Get row of B matrix and 2D matrix of C tensor #
    #-----------------------------------------------#
    B_rows , C_matrices = [] , []
    for k in [0,1]:
        Bk, Dk = wilson_auxlinB(m,o,n,k)
        Ck = wilson_auxlinC(m,o,n,k,Bk,Dk)
        # In old Cartesian coord. systems
        Bk = np.matrix(Bk) * Tbar.transpose()
        Ck = Tbar * Ck * Tbar.transpose()
        # Append data
        B_rows.append(np.array(Bk.tolist()[0]))
        C_matrices.append(Ck)

    #------------------------------------------#
    # Consider all the atoms to create B and C #
    #------------------------------------------#
    final_B = []
    for Bk in B_rows:
        fBk = [0.0 for idx in range(3*natoms)]
        for a in [M,O,N]:
            for i in [0,1,2]:
                if a == M: fBk[3*M+i] = Bk[0+i]
                if a == O: fBk[3*O+i] = Bk[3+i]
                if a == N: fBk[3*N+i] = Bk[6+i]
        final_B.append(fBk)

    final_C = []
    for Ck in C_matrices:
        #fCk = np.zeros( (3*natoms,3*natoms) )
        fCk = [ [0.0 for idx1 in range(3*natoms)] for idx2 in range(3*natoms)]
        for a in [M,O,N]:
            for i in [0,1,2]:
                # Select row in C
                if a == M: row = 0+i
                if a == O: row = 3+i
                if a == N: row = 6+i
                for b in [M,O,N]:
                    for j in [0,1,2]:
                        # Select col in C
                        if b == M: col = 0+j
                        if b == O: col = 3+j
                        if b == N: col = 6+j
                        fCk[3*a+i][3*b+j] = Ck[row,col]
        final_C.append(fCk)

    return final_B, final_C
#-----------------------------------------------#
def wilson_torsion(bond_vectors,ijkl,natoms):
    '''
    Returns the row of the B matrix associated to the
    i-j-k-l torsion and the corresponding C matrix.
    Check ref [1] and [5].
    '''

    # Using nomenclature of reference [5]
    m, o, p, n = ijkl
    u, lambda_u = bond_vectors[(o,m)]
    v, lambda_v = bond_vectors[(p,n)]
    w, lambda_w = bond_vectors[(o,p)]

    uxw = np.cross(u,w)
    vxw = np.cross(v,w)

    cosPhi_u = np.dot(u,w) / np.linalg.norm(u) / np.linalg.norm(w)
    sinPhi_u = np.sqrt(1.0 - cosPhi_u**2)
    cosPhi_v = -np.dot(v,w) / np.linalg.norm(v) / np.linalg.norm(w)
    sinPhi_v = np.sqrt(1.0 - cosPhi_v**2)

    # Get internal coordinate: dihedral angle
    cosq = np.dot(uxw,vxw) / sinPhi_u / sinPhi_v
    if   abs(cosq - 1.0) < EPS_SCX:
       cosq = +1.0; q = 0.0
    elif abs(cosq + 1.0) < EPS_SCX:
       cosq = -1.0; q = np.pi
    else: q = np.arccos(cosq)

    # Calculating 1st derivatives: row of B matrix #
    B_row = [0.0 for idx in range(3*natoms)]
    for a in [m,o,p,n]:
        # Get zeta values
        if a == m: zeta_amo = +1.0; zeta_apn =  0.0; zeta_aop =  0.0
        if a == o: zeta_amo = -1.0; zeta_apn =  0.0; zeta_aop = +1.0
        if a == p: zeta_amo =  0.0; zeta_apn = +1.0; zeta_aop = -1.0
        if a == n: zeta_amo =  0.0; zeta_apn = -1.0; zeta_aop =  0.0
        for i in [0,1,2]:
            # Get B element
            dq_dai =  zeta_amo * uxw[i] / lambda_u / sinPhi_u / sinPhi_u + \
                      zeta_apn * vxw[i] / lambda_v / sinPhi_v / sinPhi_v + \
                      zeta_aop * uxw[i] * cosPhi_u / lambda_w / sinPhi_u / sinPhi_u + \
                      zeta_aop * vxw[i] * cosPhi_v / lambda_w / sinPhi_v / sinPhi_v
            B_row[3*a+i] = dq_dai
    # Calculating 2nd derivatives: 2D matrix of C tensor #
    #C_matrix = np.zeros( (3*natoms,3*natoms) )
    C_matrix = [ [0.0 for idx1 in range(3*natoms)] for idx2 in range(3*natoms)]
    for a in [m,o,p,n]:
      for i in [0,1,2]:
        for b in [m,o,p,n]:
          for j in [0,1,2]:
            if C_matrix[3*a+i][3*b+j] != 0.0: continue
            # Define all delta and zeta values
            if a == m: zeta_amo = +1.0; zeta_anp =  0.0; zeta_apo =  0.0; zeta_aop =  0.0; zeta_ano =  0.0
            if a == o: zeta_amo = -1.0; zeta_anp =  0.0; zeta_apo = -1.0; zeta_aop = +1.0; zeta_ano = -1.0
            if a == p: zeta_amo =  0.0; zeta_anp = -1.0; zeta_apo = +1.0; zeta_aop = -1.0; zeta_ano =  0.0
            if a == n: zeta_amo =  0.0; zeta_anp = +1.0; zeta_apo =  0.0; zeta_aop =  0.0; zeta_ano = +1.0

            if b == m: zeta_bom = -1.0; zeta_bmo = +1.0; zeta_bnp =  0.0; zeta_bop =  0.0; zeta_bpo =  0.0
            if b == o: zeta_bom = +1.0; zeta_bmo = -1.0; zeta_bnp =  0.0; zeta_bop = +1.0; zeta_bpo = -1.0
            if b == p: zeta_bom =  0.0; zeta_bmo =  0.0; zeta_bnp = -1.0; zeta_bop = -1.0; zeta_bpo = +1.0
            if b == n: zeta_bom =  0.0; zeta_bmo =  0.0; zeta_bnp = +1.0; zeta_bop =  0.0; zeta_bpo =  0.0
            zeta_bpn = - zeta_bnp

            if a == b: delta_ab = 1.0
            else:      delta_ab = 0.0
            # Get second derivative
            t01 = uxw[i]*(w[j]*cosPhi_u-u[j])/(lambda_u**2)/(sinPhi_u**4)
            t02 = uxw[j]*(w[i]*cosPhi_u-u[i])/(lambda_u**2)/(sinPhi_u**4)
            t03 = vxw[i]*(w[j]*cosPhi_v+v[j])/(lambda_v**2)/(sinPhi_v**4)
            t04 = vxw[j]*(w[i]*cosPhi_v+v[i])/(lambda_v**2)/(sinPhi_v**4)
            t05 = uxw[i]*(w[j]-2*u[j]*cosPhi_u+w[j]*cosPhi_u**2)/(2*lambda_u*lambda_w*sinPhi_u**4)
            t06 = uxw[j]*(w[i]-2*u[i]*cosPhi_u+w[i]*cosPhi_u**2)/(2*lambda_u*lambda_w*sinPhi_u**4)
            t07 = vxw[i]*(w[j]+2*v[j]*cosPhi_v+w[j]*cosPhi_v**2)/(2*lambda_v*lambda_w*sinPhi_v**4)
            t08 = vxw[j]*(w[i]+2*v[i]*cosPhi_v+w[i]*cosPhi_v**2)/(2*lambda_v*lambda_w*sinPhi_v**4)
            t09 = uxw[i]*(u[j]+u[j]*cosPhi_u**2-3*w[j]*cosPhi_u+w[j]*cosPhi_u**3) / (2*lambda_w**2*sinPhi_u**4)
            t10 = uxw[j]*(u[i]+u[i]*cosPhi_u**2-3*w[i]*cosPhi_u+w[i]*cosPhi_u**3) / (2*lambda_w**2*sinPhi_u**4)
            t11 = vxw[i]*(v[j]+v[j]*cosPhi_v**2+3*w[j]*cosPhi_v-w[j]*cosPhi_v**3) / (2*lambda_w**2*sinPhi_v**4)
            t12 = vxw[j]*(v[i]+v[i]*cosPhi_v**2+3*w[i]*cosPhi_v-w[i]*cosPhi_v**3) / (2*lambda_w**2*sinPhi_v**4)
            if i != j and a != b:
               k = [0,1,2]; k.remove(i); k.remove(j); k = k[0]
               t13 = (w[k]*cosPhi_u-u[k]) / lambda_u / lambda_w / sinPhi_u**2
               t14 = (w[k]*cosPhi_v+v[k]) / lambda_v / lambda_w / sinPhi_v**2
            else:
               t13 = 0.0
               t14 = 0.0
            dr_daidbj = zeta_amo*zeta_bmo*(t01 + t02) + \
                        zeta_anp*zeta_bnp*(t03 + t04) + \
                       (zeta_amo*zeta_bop+zeta_apo*zeta_bom)*(t05 + t06) + \
                       (zeta_anp*zeta_bpo+zeta_aop*zeta_bpn)*(t07 + t08) +\
                        zeta_aop*zeta_bpo*(t09 + t10) + \
                        zeta_aop*zeta_bop*(t11 + t12) + \
                       (zeta_amo*zeta_bop+zeta_apo*zeta_bom)*(1-delta_ab)*(j-i) / (-2.)**(abs(j-i))*t13 +\
                       (zeta_anp*zeta_bpo+zeta_aop*zeta_bpn)*(1-delta_ab)*(j-i) / (-2.)**(abs(j-i))*t14
            # Save data in both positions
            C_matrix[3*a+i][3*b+j] = dr_daidbj
            C_matrix[3*b+j][3*a+i] = dr_daidbj
    return [B_row], [C_matrix]
#-----------------------------------------------#
def wilson_getBC(xcc,all_ics):
    numics = count_ics(all_ics)
    ics_st,ics_ab,ics_lb,ics_it,ics_pt = unmerge_ics(all_ics)
    # unit bond vectors
    natoms = howmanyatoms(xcc)
    bvecs  = wilson_bvecs(xcc)
    # initialize matrices
    wilsonB = [[0.0 for row in range(numics)] for col in range(3*natoms)]
    wilsonB, wilsonC = [], []
    # B and C: stretchings
    for atoms in ics_st:
        row_B, matrix_C = wilson_stretch(bvecs,atoms,natoms)
        wilsonB += row_B
        wilsonC += matrix_C
    # B and C: angular bendings
    for atoms in ics_ab:
        row_B, matrix_C = wilson_abend(bvecs,atoms,natoms)
        wilsonB += row_B
        wilsonC += matrix_C
    # B and C: linear  bendings
    for atoms in ics_lb:
        i,j,k = atoms
        r_m = np.array(xyz(xcc,i))
        r_o = np.array(xyz(xcc,j))
        r_n = np.array(xyz(xcc,k))
        row_B, matrix_C = wilson_lbend(r_m,r_o,r_n,atoms,natoms)
        wilsonB += row_B
        wilsonC += matrix_C
    # B and C: torsions (proper and improper)
    for atoms in ics_it+ics_pt:
        row_B, matrix_C = wilson_torsion(bvecs,atoms,natoms)
        wilsonB += row_B
        wilsonC += matrix_C
    return np.matrix(wilsonB), [np.matrix(ll) for ll in wilsonC]
#-----------------------------------------------#
def wilson_getu(masses):
    '''
       masses: N
       u     : 3Nx3N
    '''
    natoms = len(masses)
    u = np.matrix(np.zeros((3*natoms,3*natoms)))
    for at in range(natoms):
        tt = 1.0/masses[at]
        ii,jj,kk = 3*at+0,3*at+1, 3*at+2
        u[ii,ii] = tt
        u[jj,jj] = tt
        u[kk,kk] = tt
    return u
#-----------------------------------------------#
def wilson_getG(u,B):
    '''
    B     : Fx3N
    G     : FxF
    -------------
    G L  = L Lambda
    G    = L Lambda L^T
    G^-1 = L^T^-1 Lambda^-1 L^-1 = L Lambda^-1 L^T
    -------------
    notice that
       L L^T = L^T L = I
    '''
    # Get G
    G = B*u*B.transpose()
    # Get inverse
    if False:
       # a) eigenvalues, eigenvectors
       Lambda, L = np.linalg.eigh(G)
       # b) reorder
       idxs = sorted([(abs(li),idx) for idx, li in enumerate(Lambda)])
       idxs = [idx for li,idx in idxs]
       Lambda = [Lambda[idx] for idx in idxs]
       L      = L[:,idxs]
       # Generalized inverse (not inverse of zero eigenvalues)
       G    = L * np.diag(Lambda) * L.transpose()
       Lambda_inv = [1.0/li if li > EPS_GIV else li for li in Lambda]
       Ginv2 = L * np.diag(Lambda_inv) * L.transpose()
    if True:
       Ginv = scipy.linalg.pinv(G,atol=EPS_GIV)
    # return
    return G, Ginv
#-----------------------------------------------#
def wilson_gf_internal(u,B,C,Ginv,gcc,Fcc):
    nIC, n3N = B.shape
    if gcc.shape != (n3N,  1): exit("Wrong shape for gcc")
    if Fcc.shape != (n3N,n3N): exit("Wrong shape for Fcc")
    A = u * B.transpose() * Ginv
    g = A.transpose() * gcc
    f = A.transpose() * Fcc * A
    for i in range(nIC):
        f -=  float(g[i].item()) * A.transpose() * np.matrix(C[i]) * A
    return g, f, A
#-----------------------------------------------#
def wilson_gf_nonred(G,Ginv,g,f):
    nIC, nIC = G.shape
    if g.shape != (nIC,  1): exit("Wrong shape for g")
    if f.shape != (nIC,nIC): exit("Wrong shape for f")
    P   = G * Ginv
    gnr = P * g
    fnr = P * f * P
    return gnr, fnr
#-----------------------------------------------#
def wilson_prj_rc(gnr,fnr,G,nics):
    ''' project out the reaction coordinate'''
    I   = np.identity(nics)
    p   = gnr * gnr.transpose() / (gnr.transpose() * G * gnr)
    fnr = (I - p*G) * fnr * (I-G*p)
    return fnr
#-----------------------------------------------#
def wilson_evecsincart(L,G,A,masses):
    ''' evectors in cartesian (mass-scaled)'''
    nat = len(masses)
    n3N, nIC = A.shape
    # masses 3N
    m3N = [masses[at] for at in range(nat) for i in range(3)]
    # single value decomposition for L^-1
    u, s, v = np.linalg.svd(L,full_matrices=True,compute_uv=True)
    s_inv = np.diag([1.0/s_i if s_i > EPS_SVD else s_i for s_i in s])
    L_inv = v.transpose() * s_inv.transpose() * u.transpose()
    # Get C matrix
    C = L_inv * G * L_inv.transpose()
    # Get W
    W = np.zeros( C.shape,dtype=complex )
    for i in range(len(C)): W[i,i] = cmath.sqrt(C[i,i])
   #for i in range(len(C)): W[i,i] = np.sqrt(C[i,i])
    # Get chi
    chi = A * L * W
    # Get normal-mode eigenvectors in mass-scaled cartesian
    Lcc  = np.zeros( (3*nat,nIC) )
    for j in range(nIC):
        cocient = sum([m3N[k]*chi[k,j]**2 for k in range(n3N)])
        for i in range(n3N):
            Lcc[i,j] = (np.sqrt(m3N[i])*chi[i,j]/cocient).real
    return Lcc
#-----------------------------------------------#
def calc_icfreqs(Fcc,masses,xcc,gcc,all_ics,bool_prc=False):
    '''
    As described in J. Phys. Chem. A 1998, 102, 242-247
    bool_prc: project reaction coordinate?
    '''
    # some dimensions
    nat = len(masses)
    n3N = 3 * nat
    nIC = count_ics(all_ics)
    # number of vibrational degrees of freedom
    linear = islinear(xcc)
    if linear: nvdof = n3N - 5
    else     : nvdof = n3N - 6
    if bool_prc: nvdof -= 1
    # the reduced mass (1 amu)
    mu = 1.0/AMU
    # correct shapes:  gcc (3Nx1), Fcc (3Nx3N)
    if bool_prc: gcc = np.matrix(gcc).transpose()
    else       : gcc = np.matrix(np.zeros(n3N)).transpose()
    if len(Fcc) != 3*nat: Fcc = lowt2matrix(Fcc)
    Fcc = np.matrix(Fcc)
    # 1. Calculate B matrix and C^i tensor
    B, C = wilson_getBC(xcc,all_ics)
    # 2. Calculate G (of h in other paper) and Ginv
    u       = wilson_getu(masses)
    G, Ginv = wilson_getG(u,B)
    # 3. Calculate gradient and Hessian in rics
    g,f,A = wilson_gf_internal(u,B,C,Ginv,gcc,Fcc)
    # 4.1 Project to nrics
    gnr, fnr = wilson_gf_nonred(G,Ginv,g,f)
    # 4.2 Project reaction coordinate
    if bool_prc: fnr = wilson_prj_rc(gnr,fnr,G,nIC)
    # 5.1 Eigenvalues and eigenvectors
    Lambda, L = np.linalg.eig(G*fnr)
    # 5.2 Remove imaginary part and reorder
    Lambda  = [mu*li.real for li in Lambda]
    icfreqs = [eval2afreq(li,mu) for li in Lambda]
    # 6.1 Transform eigenvecs to mass-scaled Cartesians
    L = wilson_evecsincart(L,G,A,masses)
    # 6.2 save as a list of eigenvectors
    nr,nc = L.shape
    evecs = [L[:,idx].tolist() for idx in range(nc)]
    # 7.1 indices by abs value
    idxs    = sorted([(abs(fq),fq,idx) for idx, fq in enumerate(icfreqs)])
    idxs.reverse()
    # 7.2 keep only the biggest ones (up to the number of degrees of freedom)
    idxs = idxs[:nvdof]
    # 7.3 sort again, imaginary first, then from small to big
    idxs = sorted([(fq,idx) for fqabs,fq,idx in idxs])
    # 7.4 Remove the zero eigenvalues (freqs < 0.1 cm^-1)
    idxs = [(fq,idx) for fq,idx in idxs if abs(afreq2cm(fq)) > EPS_ICF ]
    # 7.5 Prepare lists
    idxs    = [idx for fq,idx in idxs]
    Lambda  = [Lambda[idx]                   for idx in idxs]
    icfreqs = [icfreqs[idx]                  for idx in idxs]
    evecs   = [evecs[idx]                    for idx in idxs]
    # return data
    return icfreqs, Lambda, evecs
#-----------------------------------------------#
def nonredundant(xcc, masses, gcc, Fcc, all_ics, ccfreqs,
                  unremov=None, ncycles=None, extra=0):
    '''
    Reduces a redundant set of internal coordinates to a non-redundant
    one by removing coordinates one at a time (in random order), as long
    as the internal-coordinate frequencies still match the Cartesian ones.

    unremov: list of ics that must NOT be removed (it is modified in place
             and is also returned implicitly through all_ics)
    '''
    if unremov is None:
        unremov = []

    mu    = 1.0 / AMU
    nvdof = len(ccfreqs) + extra
    nics  = count_ics(all_ics)

    # no hay suficientes internas
    if nics < nvdof:
        raise Exception

    def _check(ics):
        '''Devuelve True si las frecuencias con `ics` coinciden con ccfreqs'''
        try:
            icfreqs = calc_icfreqs(Fcc, masses, xcc, gcc, ics)[0]
            return same_freqs(ccfreqs, icfreqs)
        except:
            return False

    same = _check(all_ics)

    # ya es un conjunto no redundante (o de partida no es válido)
    if nics == nvdof or not same:
        return all_ics, same

    original = list(all_ics)
    count = 0

    while True:
        candidates = [ic for ic in all_ics if ic not in unremov]
        if not candidates:
            break

        random.shuffle(candidates)
        removed_one = False

        for target in candidates:
            count += 1

            all_ics.remove(target)
            if _check(all_ics):
                removed_one = True
                break                      # eliminado con éxito, se reinicia el bucle externo
            else:
                all_ics.append(target)     # se restaura
                unremov.append(target)

            if ncycles is not None and count >= ncycles:
                break

        if count_ics(all_ics) <= nvdof: break
        if ncycles is not None and count >= ncycles: break
        # ningún candidato de esta ronda se pudo eliminar -> no hay más progreso posible
        if not removed_one: break

    # mantener el orden original
    all_ics = [ic for ic in original if ic in all_ics]

    # comprobación final
    same = _check(all_ics)

    return all_ics, same
#-----------------------------------------------#
class Molecule():

      # Initialization method
      def __init__(self,label=None):

          self._label    = label

          # Unidimensional
          self._mform    = "-"
          self._mu       = None
          self._ch       = None
          self._mtp      = None
          self._V0       = None
          self._pgroup   = None
          self._rotsigma = None
          self._natoms   = None
          self._nel      = None # number of electrons
          self._rtype    = None
          self._linear   = None

          # Multi-dimensional
          self._atnums   = None
          self._symbols  = None
          self._masses   = None
          self._les      = None # list of electronic states
          self._itensor  = None
          self._imoms    = None
          self._rotTs    = None

          # Arrays of importance
          self._xcc      = None
          self._gcc      = None
          self._Fcc      = None
          self._xms      = None
          self._gms      = None
          self._Fms      = None

          # related to frequencies
          self._fscal    = 1.0
          self._nvdof    = None
          self._cczpe    = None
          self._ccfreqs  = None
          self._ccFevals = None
          self._ccFevecs = None
          self._iczpe    = None
          self._icfreqs  = None
          self._icFevals = None
          self._icFevecs = None

          # other stuff for very particular occasion
          self._gts      = None

      def __str__(self): return self._mform

      def setvar(self,xcc=None,gcc=None,Fcc=None,\
                      atonums=None,symbols=None,masses=None,\
                      ch=None,mtp=None, V0=None,\
                      pgroup=None,rotsigma=None,\
                      fscal=None,les=None):

          if xcc      is not None: self._xcc      = xcc
          if gcc      is not None: self._gcc      = gcc
          if Fcc      is not None: self._Fcc      = Fcc

          if atonums  is not None: self._atnums   = atonums
          if symbols  is not None: self._symbols  = symbols
          if masses   is not None: self._masses   = masses

          if ch       is not None: self._ch       = int(ch)
          if mtp      is not None: self._mtp      = int(mtp)
          if V0       is not None: self._V0       = V0

          if pgroup   is not None: self._pgroup   = pgroup
          if rotsigma is not None: self._rotsigma = rotsigma

          if fscal    is not None: self._fscal    = fscal
          if les      is not None: self._les      = les

      def genderivates(self):
          self._mform   = get_molformula(self._symbols)
          self._natoms  = len(self._atnums)
          self._mass    = sum(self._masses)
          self._nel     = sum(self._atnums)-self._ch
          if self._les is None: self._les = [ (self._mtp,0.0) ]

      def prepare(self):
          # check atnums
          if self._atnums is not None and type(self._atnums[0]) == str:
             self._symbols = list(self._atnums)
          # check symbols
          if self._symbols is not None and type(self._symbols[0]) == int:
             self._atnums  = list(self._symbols)
          # Get both atnums and symbols if None
          if self._atnums  is None: self._atnums  = symbols2atonums(self._symbols)
          if self._symbols is None: self._symbols = atonums2symbols(self._atnums)
          # check masses
          if self._masses is None:
             self._masses = atonums2masses(self._atnums)
          # derivated magnitudes
          self.genderivates()
          # check Fcc
          if self._Fcc not in (None,[]) and len(self._Fcc) != 3*self._natoms:
             self._Fcc = lowt2matrix(self._Fcc)


      def setup(self,mu=1.0/AMU,projgrad=False):
          self._mu = mu
          # derivated magnitudes (again, in case sth was modified)
          # for example, when set from gts and masses are added latter
          self.genderivates()
          # shift to center of mass and reorientate molecule
          idata = (self._xcc,self._gcc,self._Fcc,self._masses)
          self._xcc, self._gcc, self._Fcc = center_and_orient(*idata)
          # Generate mass-scaled arrays
          self._xms = cc2ms_x(self._xcc,self._masses,self._mu)
          self._gms = cc2ms_g(self._gcc,self._masses,self._mu)
          self._Fms = cc2ms_F(self._Fcc,self._masses,self._mu)
          #-------------#
          # Atomic case #
          #-------------#
          if self._natoms == 1:
              self._nvdof    = 0
              self._linear   = False
             #self._xms      = list(self._xcc)
             #self._gms      = list(self._gcc)
             #self._Fms      = list(self._Fcc)
              self._ccfreqs  = []
              self._ccFevals = []
              self._ccFevecs = []
          #----------------#
          # Molecular case #
          #----------------#
          else:
             # Calculate inertia
             self._itensor = get_itensor_matrix(self._xcc,self._masses)
             self._imoms, self._rotTs, self._rtype, self._linear = \
                     get_itensor_evals(self._itensor)
             # Vibrational degrees of freedom
             if self._linear: self._nvdof = 3*self._natoms - 5
             else           : self._nvdof = 3*self._natoms - 6
             # calculate frequencies
             if self._Fcc is None        : return
             if len(self._Fcc) == 0      : return
             if self._ccfreqs is not None: return

             v0   = self._gms if projgrad else None
             data = calc_ccfreqs(self._Fcc,self._masses,self._xcc,self._mu,v0=v0)
             self._ccfreqs, self._ccFevals, self._ccFevecs = data
             # Scale frequencies
             self._ccfreqs = scale_freqs(self._ccfreqs,self._fscal)


      def icfreqs(self,ics,bool_pg=False):
          #----------------#
          # Molecular case #
          #----------------#
          if self._natoms != 1:
             ituple = (self._Fcc,self._masses,self._xcc,self._gcc,ics,bool_pg)
             self._icfreqs, self._icFevals, self._icFevecs = calc_icfreqs(*ituple)
          #-------------#
          # Atomic case #
          #-------------#
          else:
             self._icfreqs  = []
             self._icFevals = []
             self._icFevecs = []
          # scale frequencies
          self._icfreqs = [freq*self._fscal for freq in self._icfreqs]
#================================#
def print_ics(ics):
    print("    num ics: %i"%len(ics))
    ics_st,ics_ab,ics_lb,ics_it,ics_pt = unmerge_ics(ics)
    for ics_i in (ics_st,ics_ab,ics_lb,ics_it,ics_pt):
        for idx in range(0,len(ics_i),6):
            line = ""
            for icatoms in ics_i[idx:idx+6]:
                sic = "-".join([str(atom+1) for atom in icatoms])
                if icatoms in ics_lb: sic += "(x2)"
                line += sic+"  "
            print("     "+line)
#--------------------------------#
def print_freqs(ccfreqs,icfreqs):
    count = 0
    for ccf,icf in zip(ccfreqs,icfreqs):
        count += 1
        ccf = get_sfreq(ccf)
        icf = get_sfreq(icf)
        diff = abs(float(ccf)-float(icf))
        line = "    (%2i) %s   %s   (cm^-1)"%(count,ccf,icf)
        if diff > EPS_CCIC: line += " <--"
        print(line)
    print()
#--------------------------------#
def print_analysis(icfreqs,ics,L2,nvdof_ps):

    # one label per B row (a linear bending contributes two rows)
    ics_rows = []
    for ictype,icatoms in ics:
        if ictype == "lb": ics_rows += [("lb_a",icatoms),("lb_b",icatoms)]
        else             : ics_rows += [(ictype,icatoms)]

    NN = 10
    for ic in ics_rows:
        ictype,icatoms = ic
        if icatoms is None: continue
        sic = ic2string(ic)
        NN = max(NN,len(sic))

    NEPL = 6
    for idx0 in range(0,len(icfreqs),NEPL):
        line0 = (" %%%is "%NN)%"int.coord."
        for icfreq in icfreqs[idx0:idx0+NEPL]:
            line0 += "| %8.2f "%afreq2cm(icfreq)
        print("-"*len(line0))
        print(line0)
        print("-"*len(line0))
        nn = len(icfreqs[idx0:idx0+NEPL])
        weights = [0.0 for ii in range(nn)]
        for idx1,(ictype,icatoms) in enumerate(ics_rows):
            if ictype == "trx1": print("-"*len(line0))
            sic  = ic2string((ictype,icatoms))
            line = (" %%-%is "%NN)%sic
            count = 0
            for idx2 in range(idx0,idx0+NEPL):
                if idx2+1 > len(L2[idx1]): continue
                line += "| %8.3f "%L2[idx1][idx2]
                if idx1 >= nvdof_ps:
                   weights[count] += L2[idx1][idx2]
                count += 1
            print(line)
        print("-"*len(line0))
        lineN = (" %%-%is "%NN)%"weight:"
        for weight in weights: lineN += "| %8.3f "%weight
        print(lineN)
        print("-"*len(line0))
        print("")
        print("")
#================================#


#================================#
#--------------------------------#
def get_sfreq(freq,ndecimals=1):
    return "%%8.%if"%ndecimals%(afreq2cm(freq))
#--------------------------------#
def get_ics(mol,dp=None):
    # Get internal coordinates
    ics = ics_from_geom(mol._xcc,mol._symbols,scale=CONNECTSCAL)
    # assert they are valid
    mol.icfreqs(ics)
    same = same_freqs(mol._ccfreqs,mol._icfreqs)
    if not same:
       print("FAIL! Unable to find valid set of internal coordinates...")
       raise Exception
    # Depure them if possible
    ics2 = ics_depure(ics)
    mol.icfreqs(ics2)
    same = same_freqs(mol._ccfreqs,mol._icfreqs)
    if same: ics = ics2
    # non-redundant
    args = (mol._xcc,mol._masses,mol._gcc,mol._Fcc,ics,mol._ccfreqs)
    ics2, same = nonredundant(*args)
    if same: ics = ics2
    else   : print("    * Non-redundant set NOT FOUND")
    # all with TS numeration
    if dp is not None:
       final_ics = []
       for ictype,icatoms in ics:
           icatoms = tuple([dp[at] for at in icatoms])
           final_ics.append( (ictype,icatoms) )
    else:
       final_ics = ics
    # return data
    return final_ics
#--------------------------------#
def center_and_orient_fragment(xcc_all,atonums_all,fragment):
    xcc     = []
    atonums = []
    for at in fragment: xcc     += xcc_all[3*at:3*at+3]
    for at in fragment: atonums += [atonums_all[at]]
    # masses of fragment
    masses = [dpt_z2m[z] for z in atonums]
    # number of atoms?
    nats = len(xcc) // 3
    if nats == 1: return xcc, None
    # use centroid or center of mass to set origin
    if True: center = get_com(xcc,masses)
    else   : center = get_centroid(xcc)
    # set origin
    xcc     = set_origin(xcc,center)
    # Is it linear?
    itensor = get_itensor_matrix(xcc,masses)
    evalsI, rotTs, rtype, linear = get_itensor_evals(itensor)
    if not linear: return xcc, None
    # Get moments and axis of inertia
    itensor = np.matrix(itensor)
    evalsI, evecsI = np.linalg.eigh(itensor)
    evecsI = np.matrix(evecsI)
    # Rotation matrix 3N x 3N
    zeros = np.zeros((3, 3))
    R= []
    for at in range(nats):
        row = []
        count = 0
        while count < at:
              row.append(zeros)
              count += 1
        row.append(evecsI)
        while count < nats-1:
              row.append(zeros)
              count += 1
        R.append(row)
    R = np.matrix(np.block(R))
    # Rotate coordinates
    xcc = np.matrix(xcc) * R
    xcc = xcc.tolist()[0]
    # Rotate gradient
    return xcc, R
#--------------------------------#
def translation_B_rows(xcc,fragment):
    nat  = len(xcc) // 3
    # translation
    b1 = np.zeros(len(xcc))
    b2 = np.zeros(len(xcc))
    b3 = np.zeros(len(xcc))
    for atom in fragment:
        b1[3*atom+0] = 1
        b2[3*atom+1] = 1
        b3[3*atom+2] = 1
    b1 /= np.linalg.norm(b1)
    b2 /= np.linalg.norm(b2)
    b3 /= np.linalg.norm(b3)
    vecs = [list(b1),list(b2),list(b3)]
    return vecs
#--------------------------------#
def rotation_B_rows(xcc, atonums, fragment, linfrag=False):
    """Rotational B-matrix rows of one fragment embedded in the TS.

    Parameters
    ----------
    xcc      : flat 3N array/list with the Cartesian coordinates of the
               WHOLE transition state (any length unit; Bohr recommended
               for consistency with the rest of the program).
    atonums  : list of N atomic numbers of the whole TS.
    fragment : list of (0-based) indices of the atoms that belong to the
               fragment to be analyzed.
    linfrag  : True if the fragment correlates with a LINEAR product.
               Automatically set to True when len(fragment) == 2.

    Returns
    -------
    B_rot : (n_rot, 3N) numpy array. Each row is a normalized rotational
            displacement vector expressed in the ORIGINAL frame (original
            center and orientation), with zeros for atoms outside the
            fragment. n_rot = 0 (single atom), 2 (linear product), or
            3 (non-linear product).
    """
    xcc_ = np.asarray(xcc, dtype=float).ravel().reshape(-1, 3)
    natoms = len(atonums)
    if xcc_.shape[0] != natoms:
        raise ValueError("xcc and atonums have inconsistent sizes.")
    frag = list(fragment)
    nk = len(frag)

    if nk == 1:                                # atomic fragment: no rotations
        return []
        #return np.zeros((0, 3 * natoms))
    if nk == 2:                                # a diatomic is always linear
        linfrag = True

    # ---- 1. geometric centroid --------------------------------------------
    centroid = xcc_[frag].mean(axis=0)

    # ---- 2. centerered fragment coordinates -------------------------------
    r_cent = xcc_[frag] - centroid                   # shape (nk, 3)

    # ---- 3. principal axes of the geometric inertia tensor ----------------
    # I = sum_a (|r_a|^2 1 - r_a^T r_a)
    I = np.sum(r_cent**2) * np.eye(3) - r_cent.T @ r_cent
    moments, R = scipy.linalg.eigh(I) # ascending; columns of R = principal axes
    r_loc = r_cent @ R          # fragment coordinates in the local frame

    # ---- 4. infinitesimal rotations about the local axes ------------------
    # (x: (0, -z, +y);  y: (+z, 0, -x);  z: (-y, +x, 0))
    x, y, z = r_loc.T
    zero = np.zeros(nk)

    b_local = {
        'x': np.column_stack([zero,  -z,  +y]),
        'y': np.column_stack([+z  ,zero,  -x]),
        'z': np.column_stack([-y  ,  +x,zero]),
    }


   #print(f"centroid: {centroid}")
   #print(f"r_cent: {r_cent}")
   #print(f"R: {R}")
   #print(f"r_loc: {r_loc}")
   #print(f"b_local[x]: {b_local['x']}")
   #print(f"b_local[y]: {b_local['y']}")
   #print(f"b_local[z]: {b_local['z']}")


    # For a linear fragment, the local x axis corresponds to the smallest
    # principal moment and hence to the molecular axis.
    axes = ('y', 'z') if linfrag else ('x', 'y', 'z')

    rows = []
    for alpha in axes:
        # Rotate displacement vectors back to the original Cartesian frame
        b_glob = b_local[alpha] @ R.T     # shape (nk, 3)
        # Embed fragment displacements in a full 3N-dimensional vector
        row = np.zeros((natoms, 3))
        row[frag] = b_glob
        row = row.ravel()
        # Normalize
        nrm = np.linalg.norm(row)
        if nrm > 1e-10: rows.append(row / nrm)
   #    print(f"b_glob[{alpha}] = {b_glob}")
   #    print(f"normalized      : {rows[-1]}")
   #print("")
   #print("")
    return rows
#--------------------------------#
def get_wilson(xcc,all_ics):
    # some dimensions
    n3N = len(xcc)
    nIC = count_ics(all_ics)
    # number of vibrational degrees of freedom
    linear = islinear(xcc)
    if linear: nvdof = n3N - 5
    else     : nvdof = n3N - 6
    # prepare ics
    # Calculate B matrix and C^i tensor
    B, C = wilson_getBC(xcc,all_ics)
    ictype,icatoms = all_ics[0]
    return B,C, all_ics
#--------------------------------#
def get_icmodes(xcc,gcc,Fcc,B,C,masses,sval=0):
    n3N = len(xcc)
    nIC = len(C[0])
    # the reduced mass (1 amu)
    mu = 1.0/AMU
    # gradient
#   gcc = np.matrix(np.zeros(n3N)).transpose()
    gcc = np.matrix(gcc).transpose()
    # hessian
    if len(Fcc) != n3N: Fcc = lowt2matrix(Fcc)
    Fcc = np.matrix(Fcc)
    # 2. Calculate G (of h in other paper) and Ginv
    u       = wilson_getu(masses)
    G, Ginv = wilson_getG(u,B)
    # 3. Calculate gradient and Hessian in rics
    A = u * B.transpose() * Ginv
    g = A.transpose() * gcc
    f = A.transpose() * Fcc * A
    for i in range(nIC):
        f -=  float(g[i].item()) * A.transpose() * np.matrix(C[i]) * A
    # 4.1 Project to nrics
    gnr, fnr = wilson_gf_nonred(G,Ginv,g,f)
    # 4.2 Project reaction coordinate (only to MEP points but TS)
    if sval != 0: fnr = wilson_prj_rc(gnr,fnr,G,nIC)
    # 5.1 Eigenvalues and eigenvectors
    Lambda, L = np.linalg.eig(G*fnr)
    # 5.2 Remove imaginary part and reorder
    Lambda  = [mu*li.real for li in Lambda]
    icfreqs = [eval2afreq(li,mu) for li in Lambda]
    L       = np.real_if_close(L,tol=1000) # tol = 1000 epsilon de maquina
    return icfreqs, Lambda, L
#================================#


#=====================================================================#
def prepare_system(data,sval=0):
    mol = Molecule()
    mol.setvar(xcc=data["xcc"])
    mol.setvar(gcc=data["gcc"])
    mol.setvar(Fcc=data["Fcc"])
    mol.setvar(atonums=data["atonums"])
    mol.setvar(ch=data["ch"])
    mol.setvar(mtp=data["mtp"])
    mol.setvar(V0=data["V0"])
    mol.prepare()
    if sval == 0: mol.setup(projgrad=False)
    else        : mol.setup(projgrad=True)
    return mol
#---------------------------------------------------------------------#
def plot_weights(omegas, ntrarot, plotfile, max_freqs=42):

    # Remove the imaginary frequency and order all real modes by frequency
    modes = sorted(
        ((weight, float(freq.split("#")[0])) for freq,weight in omegas.items() if float(freq.split("#")[0]) >= 0.0),
        key=lambda mode: mode[1],
    )

    if not modes:
        raise ValueError("No real frequencies were found.")

    # Modes ordered by decreasing weight
    by_weight = sorted(
        range(len(modes)),
        key=lambda idx: modes[idx][0],
        reverse=True,
    )

    # The ntrarot - 1 modes with greatest weight must always be plotted
    n_red = min(max(ntrarot - 1, 0), len(modes))
    red = set(by_weight[:n_red])

    # Add the remaining modes by decreasing weight up to max_freqs
    n_selected = max(max_freqs, n_red)
    selected = set(by_weight[:n_selected])

    # Selected indices are already frequency ordered because modes is sorted
    selected = sorted(selected)

    # Insert one None for each omitted internal frequency interval
    rows = []

    for previous, current in zip([None] + selected[:-1], selected):
        if previous is not None and current > previous + 1:
            rows.append(None)

        rows.append(current)

    labels = []
    bar_y = []
    values = []
    colors = []

    for y, idx in enumerate(rows):

        if idx is None:
            labels.append("...")
            continue

        weight, freq = modes[idx]

        labels.append("%.0f"%freq)
        bar_y.append(y)
        values.append(weight)
        colors.append("goldenrod" if idx in red else "black")

    ypos = np.arange(len(rows))

    with plt.rc_context({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 14,
    }):
        height = max(4.6, 0.30 * len(rows) + 1.0)
        fig, ax = plt.subplots(figsize=(7.0, height))

        ax.barh(
            bar_y,
            values,
            height=0.72,
            color=colors,
        )

        for y, weight in zip(bar_y, values):
            ax.text(
                weight + 0.02,
                y,
                f"{weight:.3f}",
                va="center",
                ha="left",
                fontsize=13,
            )

        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)

        ax.set_xlim(0.0, 1.15)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xticks(np.arange(0.0, 1.01, 0.2))

        ax.set_xlabel(r"$\Omega_{\rm i}$", fontsize=16)
        ax.set_ylabel(
            r"Frequency (cm$^{-1}$)",
            fontsize=15,
            fontweight="bold",
        )

        ax.tick_params(
            direction="out",
            top=False,
            right=False,
        )

        fig.tight_layout()

    fig.savefig(plotfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
#=====================================================================#




#=====================================================================#
#==================== FUNCTIONS FOR READING FILES ====================#
#=====================================================================#
def read_input(fname):

    data = {
        "file_divsurf"  : None,
        "file_product1" : None,
        "file_product2" : None,
        "file_plot"     : "omegas.png",
        "eps_conn"      : CONNECTSCAL,
        "eps_ccic"      : EPS_CCIC,
        "linear_cutoff" : CUTOFF_LINEAR,
        "seed"          : int(time.time()),
        "stationary"    : True,
        "map_product1"  : {},
        "map_product2"  : {},
    }

    required    = ("file_divsurf","file_product1","file_product2")
    current_map = None

    with open(fname) as f: lines = f.readlines()

    for line in lines:

        # Remove comments
        line = line.split("#", 1)[0].strip()
        # Skip blank lines
        if line == "": continue

        fields = line.split()
        key    = fields[0]

        #------------------------------
        # Mapping sections
        #------------------------------
        if line.lower() == "product1:":
            current_map = data["map_product1"]
            continue

        if line.lower() == "product2:":
            current_map = data["map_product2"]
            continue

        if line.lower() == "end":
            current_map = None
            continue

        #------------------------------
        # Read mappings
        #------------------------------
        if current_map is not None:
            at_prod, at_ts = map(int, fields[:2])
            # store as 0-based indices
            if at_prod - 1 in current_map:
               raise ValueError(f"Product atom {at_prod} mapped twice in the same block")
            current_map[at_prod - 1] = at_ts - 1
            continue

        #------------------------------
        # Scalar parameters
        #------------------------------
        if   key == "eps_conn"     : data["eps_conn"     ] = float(fields[1])
        elif key == "eps_ccic"     : data["eps_ccic"     ] = float(fields[1])
        elif key == "linear_cutoff": data["linear_cutoff"] = float(fields[1])
        elif key == "seed"         : data["seed"         ] = int(fields[1])
        elif key == "file_plot"    : data["file_plot"    ] = fields[1]


        #------------------------------
        # Boolean parameters
        #------------------------------
        elif key == "stationary":
            if   fields[1].lower() in ("yes","y","true" ,"1"): data["stationary"] = True
            elif fields[1].lower() in ("no" ,"n","false","0"): data["stationary"] = False
            else: raise ValueError(f"Unknown value for stationary: {fields[1]}")

        #------------------------------
        # File names
        #------------------------------
        elif key in required:
            data[key] = fields[1]
        else:
            raise ValueError(f"Unknown keyword: {key}")

    #--------------------------------------
    # Basic validation
    #--------------------------------------
    missing = [k for k in required if data[k] is None]

    if missing:
       raise ValueError( "Missing required keywords: " + ", ".join(missing))

    return data
#---------------------------------------------------------------------#
def extract_lines(lines,key1,key2):
    '''
    Returns the lines between the one starting with key1 and the one starting with key2
    if fmatch is True, returns the first match
    '''
    record   = False
    selected = []
    for line in lines:
        line = line.split("#")[0].strip()
        if not line.endswith("\n"): line += "\n"
        if line.startswith(key2): record = False
        if record               : selected.append(line)
        if line.startswith(key1): record,selected = True,[]
    return selected
#---------------------------------------------------------------------#
def read_gtsfile(gtsfile):
    with open(gtsfile,'r') as ff: lines = ff.readlines()

    # Read cartesian coordinates
    info_cc = extract_lines(lines,"start_cc","end_cc")
    xcc,atonums = [],[]
    for line in info_cc:
        atonum,xx,yy,zz = line.split()
        xcc += [float(xx),float(yy),float(zz)]
        atonums.append(int(atonum))
    # Read basic information
    info_basic = extract_lines(lines,"start_basic","end_basic")
    ch, mtp, E, pgroup, rotsigma = None, None, None, None, None
    for line in info_basic:
        if line.startswith("pointgroup"):   pgroup   =       line.split()[1]
        if line.startswith("charge"):       ch       =   int(line.split()[1])
        if line.startswith("multiplicity"): mtp      =   int(line.split()[1])
        if line.startswith("rotsigma"):     rotsigma =   int(line.split()[1])
        if line.startswith("energy"):       E        = float(line.split()[1])
    # Read cartesian gradient
    info_grad = extract_lines(lines,"start_grad","end_grad")
    gcc = []
    for line in info_grad:
        gx , gy , gz = line.split()
        gcc += [float(gx),float(gy),float(gz)]
    # Read Hessian matrix
    info_hess = extract_lines(lines,"start_hess","end_hess")
    Fcc = []
    for line in info_hess:
        Fcc += [float(Fij) for Fij in line.split()]
    # Return data
    data = {"atonums":atonums , "xcc":xcc , "gcc":gcc , "Fcc":Fcc ,\
            "mtp":mtp , "ch":ch , "V0":E}
    return data
#=====================================================================#


#=====================================================================#
def deal_with_fragment(gtsfile,mapping):
    print(f"  - filename   : {gtsfile}")
    fragment = prepare_system(read_gtsfile(gtsfile))
    N        = fragment._natoms
    print(f"  - num atoms  : {N}")
    print(f"  - mol formula: {fragment._mform}")
    for idx,s in enumerate(fragment._symbols):
        x,y,z = fragment._xcc[3*idx:3*idx+3]
        print("    [%2i] %2s   %+10.5f %+10.5f %+10.5f  [%2i in TS]"%(idx+1,s,x,y,z,mapping[idx]+1))
    print(f"  - linear product?",fragment._linear)
    print(f"  - number vib. dofs:",fragment._nvdof)
    if N == 1:
       ics = []
    else:
       print(f"  - Cartesian coordinate frequencies (1/cm):")
       for idx in range(0,len(fragment._ccfreqs),5):
           print(" "*4+" ".join( [get_sfreq(f) for f in fragment._ccfreqs[idx:idx+5]]))
       print(f"  - Internal coordinates (using TS numeration)")
       ics     = get_ics(fragment,mapping)
       # print ics
       print_ics(ics)
    print("")
    return ics,list(mapping.values()),fragment._nvdof,fragment._linear
#---------------------------------------------------------------------#
def deal_with_ts(gtsfile,ats1,ats2,sval=0):
    print(f"  - filename   : {gtsfile}")
    ts = prepare_system(read_gtsfile(gtsfile),sval=sval)
    N  = ts._natoms
    print(f"  - num atoms  : {N}")
    print(f"  - mol formula: {ts._mform}")
    for idx,s in enumerate(ts._symbols):
        x,y,z = ts._xcc[3*idx:3*idx+3]
        if   idx in ats1: nfrag = 1
        elif idx in ats2: nfrag = 2
        else            : raise Exception
        print("    [%2i] %2s   %+10.5f %+10.5f %+10.5f  [fragment %i]"%(idx+1,s,x,y,z,nfrag))
    print(f"  - linear?",ts._linear)
    print(f"  - number vib. dofs:",ts._nvdof)
    if sval == 0:
       print(f"  - Cartesian coordinate frequencies (1/cm):")
    else:
       print(f"  - Cartesian coordinate frequencies (1/cm) [after projecting gradient]:")

    for idx in range(0,len(ts._ccfreqs),5):
        print(" "*4+" ".join( [get_sfreq(f) for f in ts._ccfreqs[idx:idx+5]]))
    print("")
    return ts
#---------------------------------------------------------------------#
def print_b_elements(symbols,brows):
    if len(brows) == 0:
       print("    Atomic fragment, no rotation associated!")
       return

    lines = {idx:f"    {s:2s} |  " for idx,s in enumerate(symbols)}

    for idx,symbol in enumerate(symbols):
        for brow in brows:
            lines[idx] += "  ".join(["%+6.3f"%bij for bij in brow[3*idx:3*idx+3]])
            lines[idx] += "  |  "
    for idx in range(len(symbols)): print(lines[idx])
#=====================================================================#

#=====================================================================#
def check_orthogonality(Bfinal,ntrarot,eps=1e-6):
    '''
    Scalar products between the rows of Bfinal (rows must be normalized).
    Rows are ordered as: vib (nvib), tra (6), rot (ntrarot-6).
    For each block (vib-vib, tra-tra, rot-rot, vib-tra, vib-rot, tra-rot)
    prints the pairs with |product| > eps
    '''
    # check rank
    rank = np.linalg.matrix_rank(Bfinal)
    print("  shape =", Bfinal.shape)
    print("  rank  =", rank)
    print("  full rank =", rank == min(Bfinal.shape))
    print("")

    # get number of rows of each type
    nrows = np.matrix(Bfinal).shape[0]
    nvib  = nrows - ntrarot
    ntra  = 6
    nrot  = ntrarot - ntra
    # indices of the rows of each type
    idxs = {}
    idxs["vib"] = list(range(0        ,nvib ))
    idxs["tra"] = list(range(nvib     ,nvib+ntra))
    idxs["rot"] = list(range(nvib+ntra,nrows))
    # label of each row
    labels  = ["vib%i"%(i+1) for i in range(nvib)]
    labels += ["tra%i"%(i+1) for i in range(ntra)]
    labels += ["rot%i"%(i+1) for i in range(nrot)]
    # get scalar product between each pair of vectors
    Bnorm = np.matrix([np.array(row)/np.linalg.norm(row) for row in np.array(Bfinal).tolist()])
    M     = Bnorm*Bnorm.transpose()
    # print numbers
    print(f"  number of vibrational   rows: {nvib}")
    print(f"  number of translational rows: {ntra}")
    print(f"  number of rotational    rows: {nrot}")
    print("")
    print(f"  checking orthogonality between rows using eps = {eps:.1E}:")
    print("")
    # analyze each block
    blocks = [("vib","vib"),("tra","tra"),("rot","rot"),\
              ("vib","tra"),("vib","rot"),("tra","rot")]
    for type1,type2 in blocks:
        nonzero = []
        for i in idxs[type1]:
            for j in idxs[type2]:
                # same type --> only upper triangle (diagonal excluded)
                if type1 == type2 and j <= i: continue
                if abs(M[i,j]) > eps: nonzero.append( (i,j,M[i,j]) )
        # print results for this block
        if len(nonzero) == 0:
           print("    %s-%s: all zero"%(type1,type2))
        else:
           print("    %s-%s: %i non-zero products"%(type1,type2,len(nonzero)))
          #for i,j,Mij in nonzero:
          #    print("       %-6s %-6s %+8.1E"%(labels[i],labels[j],Mij))
    print("")
#---------------------------------------------------------------------#
def check_mapping(dataINP):
    '''
    Checks the atom mapping between the product fragments and the TS:
      (a) every atom of each product is mapped (and only those atoms)
      (b) every atom of the TS appears exactly once in the two mappings
      (c) mapped atoms correspond to the same element
    Raises an exception if any of these conditions is not fulfilled
    '''
    # atomic numbers from gts files
    atonums_ts = read_gtsfile(dataINP["file_divsurf" ])["atonums"]
    atonums_p1 = read_gtsfile(dataINP["file_product1"])["atonums"]
    atonums_p2 = read_gtsfile(dataINP["file_product2"])["atonums"]
    maps       = (("product1",dataINP["map_product1"],atonums_p1),
                  ("product2",dataINP["map_product2"],atonums_p2))
    errors = []

    # (a) atoms of each product
    for label,amap,atonums in maps:
        missing = [at+1 for at in range(len(atonums)) if at not in amap]
        outside = [at+1 for at in amap if not (0 <= at < len(atonums))]
        if missing: errors.append("%s: atom(s) not mapped: %s"%(label,missing))
        if outside: errors.append("%s: atom(s) not in the gts file: %s"%(label,outside))

    # (b) atoms of the TS
    ts_atoms = list(dataINP["map_product1"].values()) + \
               list(dataINP["map_product2"].values())
    repeated = sorted(set([at+1 for at in ts_atoms if ts_atoms.count(at) > 1]))
    missing  = [at+1 for at in range(len(atonums_ts)) if at not in ts_atoms]
    outside  = sorted(set([at+1 for at in ts_atoms if not (0 <= at < len(atonums_ts))]))
    if repeated: errors.append("TS: atom(s) mapped more than once: %s"%repeated)
    if missing : errors.append("TS: atom(s) not mapped: %s"%missing)
    if outside : errors.append("TS: atom(s) not in the gts file: %s"%outside)

    # (c) same element
    for label,amap,atonums in maps:
        for at_p,at_ts in sorted(amap.items()):
            if not (0 <= at_p  < len(atonums   )): continue
            if not (0 <= at_ts < len(atonums_ts)): continue
            if atonums[at_p] != atonums_ts[at_ts]:
               errors.append("%s: atom %i (Z=%i) mapped to TS atom %i (Z=%i)"%\
                             (label,at_p+1,atonums[at_p],at_ts+1,atonums_ts[at_ts]))

    # print errors (if any)
    if len(errors) != 0:
       print("   - ERROR in the atom mapping:")
       for error in errors: print("       * %s"%error)
       print("")
       raise Exception
    print("   - atom mapping       : OK")
#=====================================================================#

#=====================================================================#
def linear_bendings_at_ts(xcc,ics,atoms_p1,atoms_p2,maxangle=CUTOFF_LINEAR):
    '''
    At the TS geometry, the linear bending of a linear product with
    a single linear bending (e.g. CO2) is replaced by the bond angle
    if the angle is smaller than maxangle. The fragment then needs
    an extra row: the rigid rotation around its axis.
    Returns the new list of ics and the list of those fragments
    '''
    # number of linear bendings in each fragment
    count = {1:0, 2:0}
    for ictype,icatoms in ics:
        if ictype != "lb": continue
        if icatoms[1] in atoms_p1: count[1] += 1
        else                     : count[2] += 1
    # replace linear bendings
    new_ics, axial = [], []
    for ictype,icatoms in ics:
        if ictype == "lb":
           if icatoms[1] in atoms_p1: K, frag = 1, atoms_p1
           else                     : K, frag = 2, atoms_p2
           m, o, n = icatoms
           theta   = np.degrees(angle(xyz(xcc,m),xyz(xcc,o),xyz(xcc,n)))
           if count[K] == 1 and theta < maxangle:
              new_ics.append( ("ab",icatoms) )
              axial.append( (frag,icatoms) )
              continue
        new_ics.append( (ictype,icatoms) )
    # keep the order used in wilson_getBC
    new_ics = merge_ics(*unmerge_ics(new_ics))
    return new_ics, axial
#=====================================================================#

#=====================================================================#
def classify_modes(fprod1,fprod2,fts,map1,map2,plotfile="omegas.png",sval=0):

    if type(sval) == float and sval == 0.0: sval = 0

    # Deal with product 1
    print(f"* FRAGMENT 1")
    ics1,atoms_p1,nvdof1,linear1 = deal_with_fragment(fprod1,map1)

    # Deal with product 2
    print(f"* FRAGMENT 2")
    ics2,atoms_p2,nvdof2,linear2 = deal_with_fragment(fprod2,map2)

    # Deal with transition state
    print("* DIVIDING SURFACE")
    ts      = deal_with_ts(fts,atoms_p1,atoms_p2,sval=sval)

    if sval != 0 and np.linalg.norm(ts._gcc) < 5e-6:
       print("    ERROR: 'stationary no' but the gradient is (numerically) zero;")
       print("           use 'stationary yes' for a stationary point")
       print("")
       raise Exception

    ccfreqs = list(ts._ccfreqs)
    ntrarot = 0

    ics_trarot = []

    print("* Translational vectors of fragment 1...")
    btra1       = translation_B_rows(ts._xcc,atoms_p1)
    ntrarot    += len(btra1)
    ics_trarot += [("trx1",None),("try1",None),("trz1",None)]
    print_b_elements(ts._symbols,btra1)
    print("")

    print("* Translational vectors of fragment 2...")
    btra2    = translation_B_rows(ts._xcc,atoms_p2)
    ntrarot += len(btra2)
    ics_trarot += [("trx2",None),("try2",None),("trz2",None)]
    print_b_elements(ts._symbols,btra2)
    print("")

    print("* Rotational vectors for fragment 1...")
    brot1    = rotation_B_rows(ts._xcc,ts._atnums,atoms_p1,linear1)
    ntrarot += len(brot1)
    if len(brot1) >= 2: ics_trarot += [("rotA1",None),("rotB1",None)]
    if len(brot1) == 3: ics_trarot += [("rotC1",None)]
    print_b_elements(ts._symbols,brot1)
    print("")

    print("* Rotational vectors for fragment 2...")
    brot2    = rotation_B_rows(ts._xcc,ts._atnums,atoms_p2,linear2)
    ntrarot += len(brot2)
    if len(brot2) >= 2: ics_trarot += [("rotA2",None),("rotB2",None)]
    if len(brot2) == 3: ics_trarot += [("rotC2",None)]
    print_b_elements(ts._symbols,brot2)
    print("")

    # Generate B matrix and checking orthogonality and full rank
    print("* Generating B matrix...")
    # (a) vibration-product coordinates
    ics     = merge_ics(*unmerge_ics(ics1+ics2))
    B,C,ics = get_wilson(ts._xcc,ics)
    # (b) add rototranslational displacements
    Bfinal  = np.matrix(B.tolist()+btra1+btra2+brot1+brot2)
    # (c) check orthogonality
    check_orthogonality(Bfinal,ntrarot)

    # (d) deal with linear bendings
    ics, axial = linear_bendings_at_ts(ts._xcc,ics,atoms_p1,atoms_p2)
    if len(axial) != 0:
       print("  --> Linear product is bent in the dividing-surface structure!")
       print("  --> Linear bending is replaced by angular bending + rigid rotation\n")
       # (d.1) vibrational part - linear bending replaced by angular bending
       B,C,ics = get_wilson(ts._xcc,ics)
       # (d.2) add rigid rotation
       for fragment,icatoms in axial:
           row = rotation_B_rows(ts._xcc,ts._atnums,fragment,False)[0]
           # add rigid rotation to B and to C
           B   = np.matrix(B.tolist()+[list(row)])
           C.append(np.matrix(np.zeros(C[0].shape)))
           # add rigid rotation as "vibrational"
           ics.append( ("rax",icatoms) )
       # (d.3) add rototranslational displacements
       Bfinal  = np.matrix(B.tolist()+btra1+btra2+brot1+brot2)
       # (d.4) check orthogonality again
       check_orthogonality(Bfinal,ntrarot)
    # ------------------------------- #

    # (e) complete set of coordinates
    ics = ics + ics_trarot

    # (f) complete C for rototranslational part
    nrows,ncols = C[0].shape
    for i in range(ntrarot): C.append( np.matrix(np.zeros((nrows,ncols))) )
    print("")

    # Get frequencies in internal coordinates
    print("* cc-frequencies vs ic-frequencies...")
    icfreqs, Lambda, L = get_icmodes(ts._xcc,ts._gcc,ts._Fcc,Bfinal,C,ts._masses,sval)

    # Exclude overall rotations and translations
    L = L.transpose().tolist()
    data_ic = []
    for idx in range(len(icfreqs)):
        icfreq   = icfreqs[idx]
        evalue   = Lambda[idx]
        Li       = L[idx]
        icsquare = icfreq ** 2
        data_ic.append( (icsquare,icfreq,evalue,Li) )
    data_ic.sort()
    if sval == 0: data_ic = data_ic[-ts._nvdof:]
    else        : data_ic = data_ic[-ts._nvdof+1:]

    # Sort according freq value and not according to its square
    data_ic.sort(key = lambda asdf:asdf[1])
    icfreqs = [icfreq for icfreq2,icfreq,evalue,Lm in data_ic]
    print_freqs(ts._ccfreqs,icfreqs)
    same    = same_freqs(ts._ccfreqs,icfreqs)
    if sval == 0:
       if same: print("    * SET IS VALID!!!\n")
       else   : print("    * SET IS NOT VALID!!!\n"); raise Exception
    else:
       print("    * Non-stationary point: gradient direction projected out")
       print("      |gradient| = %.2e hartree/bohr"%np.linalg.norm(ts._gcc))
       if not same:
          print("    * Note: ic- and cc-frequencies differ (expected at non-stationary points)\n")
       print("")

    print("* Analyzing modes...")
    L2 = [[Lm_i**2 for Lm_i in Lm]  for icfreq2,icfreq,evalue,Lm in data_ic]
    L2 = np.matrix(L2).transpose().tolist()

    # print table with analysis of each mode
    print_analysis(icfreqs,ics,L2,nvdof1+nvdof2)

    print(f"* Mode classification:")
    print("")
    print("      freq (cm^-1)   Omega_i  ")
    print("    --------------------------")

    omegas = {}
    evects = {}
    for icfreq2,icfreq,evalue,Lm in data_ic:
        sfreq  = get_sfreq(icfreq).strip()
        omega  = sum([Lm_i**2 for Lm_i in Lm[-ntrarot:]])
        # (quasi)degenerate modes would share the same key --> add suffix
        key, nn = sfreq, 1
        while key in omegas:
            nn += 1
            key = "%s#%i"%(sfreq,nn)
        omegas[key] = omega
        evects[key] = Lm

    # locate imaginary frequency
    ifreq = None
    if sval == 0:
       lifreq = [sfreq for sfreq,omega in omegas.items() if "-" in sfreq]
       if len(lifreq) == 1: ifreq = lifreq[0]
       else:
          print("    * WARNING: %i imaginary frequencies found (1 expected"%len(lifreq))
          print("      for a first-order saddle point)")

    # print imaginary freq
    if ifreq is not None:
       iomega = omegas[ifreq]
       print(f"       {ifreq:>8s}        ifreq   [roto-translational]")
    else:
       print(f"       {'   --   ':s}     gradient   [roto-translational]")

    count = 1
    nrototrans = ts._nvdof-nvdof1-nvdof2

    # print the rest
    tmp = [(omega,sfreq) for sfreq,omega in omegas.items() if sfreq != ifreq]
    tmp.sort(reverse=True)
    for omega,sfreq in tmp:
        count += 1
        line   = "       %8s        %5.3f"%(sfreq,omega)
        if count <= nrototrans:
           line += "   [roto-translational]"
        print(line)
    print("")

    if 1 <= nrototrans-1 < len(tmp):
       DELTAOMEGA = tmp[nrototrans-2][0] - tmp[nrototrans-1][0]
       print(f"         Delta Omega = {DELTAOMEGA:.3f}")
       print("")

    if plotfile is not None:
       print(f"- Creating plot with weights: {plotfile}")
       plot_weights(omegas,nrototrans,plotfile)
       print("")
    return omegas,evects
#---------------------------------------------------------------------#
def main():

    # Read input file
    fname, ginp = parse_args()

    print(HEADER)

    if ginp:
       fname = "portico.inp"
       print(f"   - writing example input file: {fname}")
       if os.path.exists(fname):
          print(f"   - error! file already exists!")
       else:
          with open(fname,'w') as asdf: asdf.write(INPUTEXAMPLE)
       print("")
       return

    print(f"   - input file         : {fname}")
    if not os.path.exists(fname):
       print("   - file not found!\n")
       return 
    print("")

    dataINP  = read_input(fname)

    # Print tolerance and bonding threshold
    print(rf"   - frequency tolerance: {dataINP['eps_ccic']:.2f} cm-1")
    print(rf"   - bonding threshold  : {dataINP['eps_conn']:.2f}")
    print(rf"   - linear  cutoff     : {dataINP['linear_cutoff']:.2f} degrees")
    print(rf"   - current random seed: {dataINP['seed']}")
    print(rf"   - stationary point   : {dataINP['stationary']}")
    print()

    # check mapping
    check_mapping(dataINP)
    print()

    global EPS_CCIC, CONNECTSCAL, CUTOFF_LINEAR, EPS_LINEAR
    EPS_CCIC      = dataINP['eps_ccic']
    CONNECTSCAL   = dataINP['eps_conn']
    random.seed(dataINP["seed"])

    CUTOFF_LINEAR = dataINP['linear_cutoff']
    if BOOL_LINEAR:
       EPS_LINEAR = 180.0 - CUTOFF_LINEAR
    else:
       EPS_LINEAR = 4.5


    # classify modes
    plotfile  = dataINP["file_plot"]
    fts       = dataINP["file_divsurf"]
    fprod1    = dataINP["file_product1"]
    fprod2    = dataINP["file_product2"]
    map1      = dataINP["map_product1"]
    map2      = dataINP["map_product2"]
    if dataINP["stationary"]: sval = 0
    else                    : sval = 1.0

    kwargs = {"plotfile":plotfile, "sval":sval}
    classify_modes(fprod1,fprod2,fts,map1,map2,**kwargs)
#=====================================================================#

if __name__ == '__main__': main()

