#!/usr/bin/python3

#=====================================================================#
import os
import sys
from   ase.data import atomic_numbers
#---------------------------------------------------------------------#
ANGSTROM = 0.52917721067                                              #
#=====================================================================#




#-----------------------------------------------#
def flatten_llist(list_of_lists):
    ''' converts list of lists to list'''
    flattened_list = [y for x in list_of_lists for y in x]
    return flattened_list
#---------------------------------------------#

#=======================================================#
# FUNCTIONS FOR READING THE LOG FILE OF GAUSSIAN PROG.  #
#=======================================================#
def split_gaulog_into_gaublocks(filename):
    # Key to split the system into blocks
    str_end='Normal termination'
    # Read log/out file and join lines as string
    with open(filename,'r') as asdf: lines = asdf.readlines()
    text  = "".join(lines)
    # Divide by str_end (last element has to be excluded)
    blocks = text.split(str_end)[:-1]
    # For some reason, sometimes Gaussian prints a block
    # without information. In these cases, the block consists
    # of a few lines. Here, we exclude that cases
    #print [len(block.split("\n")) for block in blocks]
    blocks = [block for block in blocks if len(block.split("\n")) > 300]
    # Remove the lines list and the whole text
    del lines, text
    # No normal termination?
    return blocks
#-------------------------------------------------------#
def get_data_from_maintext(mtext):
    key_geom1 = 'Z-Matrix orientation'
    key_geom2 = 'Input orientation'
    key_geom3 = 'Standard orientation'
    key_force = "     Forces ("
    key_zmat  = 'Final structure in terms of initial Z-matrix:'
    key_end   = "------------------"
    key_1bar  = "\\"
    key_oniom = "ONIOM: extrapolated energy"

    # (a) Find cartesian coordinates
    if   key_geom1 in mtext: geom = mtext.split(key_geom1)[-1]
    elif key_geom2 in mtext: geom = mtext.split(key_geom2)[-1]
    elif key_geom3 in mtext: geom = mtext.split(key_geom3)[-1]
    else                   : geom,xcc = None, None
    if geom is not None:
       # convert to list of lines and get the lines associated to geometry
       geom = "\n".join(geom.split("\n")[5:])
       idx  = geom.find(key_end)
       geom = geom[:idx].strip()
       # convert to list of floats
       geom = [line.split() for line in geom.split("\n") if line.strip() != ""]
       xcc  = [[float(x),float(y),float(z)] for (_,atnum,_,x,y,z) in geom]
       xcc  = flatten_llist(xcc)
    # (b) Find forces --> gradient
    if key_force in mtext:
       force = mtext.split(key_force)[-1]
       # convert to list of lines and get the lines associated to forces
       force = "\n".join(force.split("\n")[3:])
       idx = force.find(key_end)
       force = force[:idx]
       # convert to list of floats
       force = [line.split() for line in force.split("\n") if line.strip() != ""]
       gcc  = [[-float(gx),-float(gy),-float(gz)] for (_,atnum,gx,gy,gz) in force]
       gcc  = flatten_llist(gcc)
    else: gcc = None
    # (c) Find z-matrix
    if  key_zmat in mtext:
        lines_zmat = mtext.split(key_zmat)[-1].strip().split("\n")
        for idx,line in enumerate(lines_zmat):
            line = line.strip()
            if line == "" or key_1bar in line:
               idx1 = idx
               break
        zmat = [line.strip() for line in lines_zmat[:idx1] if "Variables:" not in line]
    else: zmat = None
    # (d) ONIOM energy?
    E_ONIOM = None
    for line in mtext.split("\n")[::-1]:
        if key_oniom in line:
           E_ONIOM = float(line.split()[-1])
           break
    # Convert xcc to bohr
    if xcc is not None: xcc = [xi/ANGSTROM for xi in xcc]
    # Return data
    return xcc, gcc, zmat, E_ONIOM
#-------------------------------------------------------#
def get_data_from_archive(summary):
    # Keywords to look for
    key_hag  = "#"
    key_1bar ='\\'
    key_2bar ='\\\\'
    key_ver  ='Version'
    key_en1  ='State='
    key_en2  ='RMSD='
    key_imag ='NImag='
    # logical variables
    Lzmat = False
    Lxyz  = False
    Lhess = False
    # (a) the command line
    idx1 = summary.find(key_hag,0)+len(key_hag)
    idx2 = summary.find(key_2bar,idx1)
    commands = summary[idx1:idx2]
    # (b) the comment line
    idx3 = summary.find(key_2bar,idx2)+len(key_2bar)
    idx4 = summary.find(key_2bar,idx3)
    comment = summary[idx3:idx4]
    # (c) charge and multiplicity
    idx5 = summary.find(key_2bar,idx4)+len(key_2bar)
    idx6 = summary.find(key_1bar,idx5)
    ch,mtp = [int(value) for value in summary[idx5:idx6].split(",")]
    # (d) z-matrix or Cartesian coordinates
    idx7 = summary.find(key_ver,idx6)
    geom = [string for string in summary[idx6+len(key_1bar):idx7].split(key_1bar) if string != ""]
    if len(geom[0]) <= 4:
       Lzmat   = True
       zmat    = list(geom)
       symbols = [line.split(",")[0] for line in zmat if "=" not in line]
       xcc     = None
    else:
       Lxyz    = True
       zmat    = None
       symbols = [line.split(",")[0]   for line in geom]
       # sometimes, this line has 5 elements instead of 4
       # for this reason, coordinates are extracted with [-3:]
       # instead of [1:]
       xyz = [line.split(",")[-3:] for line in geom]
       xyz = [[float(x),float(y),float(z)] for (x,y,z) in xyz]
       xcc = flatten_llist(xyz)
    # (e) Energy and other info
    idx8a = summary.find(key_ver,idx7)
    idx8b = summary.find(key_en1,idx7)
    idx8  = max(idx8a,idx8b)
    idx9  = summary.find(key_1bar,idx8)+len(key_1bar)
    idx10 = summary.find(key_en2,idx9)
    str_energies = summary[idx9:idx10].replace(key_1bar," ")
    energies = str_energies.split()
    energies = [line.split("=") for line in energies]
    # remove S**2 (for open-shell)
    energies = [(float(energy),level.strip()) for level,energy in energies if not level.strip().startswith("S2")]
    # (f) Hessian matrix
    Lhess = key_imag in summary
    if Lhess:
       idx11=summary.find(key_imag,0)+len(key_imag)
       idx12=summary.find(key_2bar,idx11)
       num_imag = int(summary[idx11:idx12])
       idx12 += len(key_2bar)
       idx13=summary.find(key_2bar,idx12)
       # low-triangle hessian
       Fcc = [float(value) for value in summary[idx12:idx13].split(",")]
    else:
       num_imag = -1
       Fcc  = None
    # (g) Gradient may appera after hessian matrix
    if Lhess:
       idx13 += len(key_2bar)
       idx14=summary.find(key_2bar,idx13)
       gcc = [float(value) for value in summary[idx13:idx14].split(",")]
    else:
       gcc = None
    # Convert xcc to bohr
    if xcc is not None: xcc = [xi/ANGSTROM for xi in xcc]
    return commands,comment,ch,mtp,symbols,xcc,gcc,Fcc,energies,num_imag,zmat
#-------------------------------------------------------#
def get_data_from_gaublock(gaublock):
    '''
    gaublock is a string 
    gaublock contains the info from begining till Normal termination
    '''
    # Divide the block into the summary part and the rest
    key_start='GINC'
    key_end  ='@'
    mtext    = gaublock.split(key_start)[0]
    summary  = gaublock.split(key_start)[1].split(key_end)[0]
    # Remove the initial blank space of each line in summary
    # Also remove the line breaks
    summary  = "".join([line.strip() for line in summary.split("\n")])
    # Data in summary (aka archive)
    commands,comment,ch,mtp,symbols,xcc,gcc,Fcc,energies,num_imag,zmat = get_data_from_archive(summary)
    # Data in main text (excluding archive)
    xcc_mt, gcc_mt, zmat_mt, E_oniom = get_data_from_maintext(mtext)
    # If data not in archive --> get from main text
    if xcc  is None and  xcc_mt is not None: xcc  =  xcc_mt
    if gcc  is None and  gcc_mt is not None: gcc  =  gcc_mt
    if zmat is None and zmat_mt is not None: zmat = zmat_mt
    # Return data
    return commands,comment,ch,mtp,symbols,xcc,gcc,Fcc,energies,E_oniom,num_imag,zmat
#-------------------------------------------------------#
def read_gaussian_log(filename):
    if not os.path.exists(filename): return
    # split lines into blocks (in case of Link1)
    blocks = split_gaulog_into_gaublocks(filename)
    # Get info of each block
    data = [get_data_from_gaublock(block) for block in blocks]
    # There is nothing to return
    if data == []: return [None]*12
    # Localize data with hessian matrix
    IDX = -1
    for idx,data_i in enumerate(data):
        Fcc =  data_i[7]
        if Fcc is not None:
           IDX = idx
           break
    # Return the best set of data (the last with the hessian or the last block)
    commands,comment,ch,mtp,symbols,xcc,gcc,Fcc,energies,E_oniom,num_imag,zmat = data[IDX]
    # If user does not ask for level, send one of lowest energy
    energies.sort()
    energy,level = energies[0]
    # oniom?
    if E_oniom is not None:
       energy = E_oniom
       level  = "ONIOM"
    # Return data
    return commands,comment,ch,mtp,symbols,xcc,gcc,Fcc,energy,num_imag,zmat,level
#-------------------------------------------------------#
#=======================================================#



#=====================================================================#
def matrix2lowt(matrix):
    if matrix is None: return None
    nrows = len(matrix)
    ncols = len(matrix[0])
    low   = []
    for row in range(nrows):
        for col in range(0,row+1):
            Fij = matrix[row][col]
            low.append(Fij)
    #import numpy as np
    #low2 = matrix[np.tril_indices(matrix.shape[0])]
    return low
#=====================================================================#


#=====================================================================#
def write_gtsfile(xcc,atonums,ch,mtp,E,pgroup,rotsigma,gcc,Fcc,gtsfile,freqs=None,level=""):
    if gcc   is None: gcc = []
    if Fcc   is None: Fcc = []
    if freqs is None: freqs = []
    nat      = len(atonums)
    str_gts  = ""
    # write level of calculation
    if level != "": str_gts += "# level: %s\n"%level
    # Write basic data
    str_gts += "# Charge, multiplicity, energy [hartree],\n"
    str_gts += "# point group and rotational symmetry number\n"
    str_gts += "start_basic\n"
    str_gts += "   charge        %i\n"%ch
    str_gts += "   multiplicity  %i\n"%mtp
    str_gts += "   energy       %-+16.8f # Total energy in hartree\n"%E
    str_gts += "   pointgroup    %-5s           # Point group\n"%pgroup
    str_gts += "   rotsigma      %-5i           # Rotational sigma\n"%rotsigma
    str_gts += "end_basic\n\n"
    # Write atomic numbers and cartesian coordinates
    str_gts += "# Atomic number and non-scaled cartesian coordinates [bohr]\n"
    str_gts += "start_cc\n"
    for at in range(nat):
        atonum = atonums[at]
        xx,yy,zz = xcc[3*at:3*at+3]
        str_gts += "   %03i   %+15.8E  %+15.8E  %+15.8E\n"%(atonum,xx,yy,zz)
    str_gts += "end_cc\n\n"
    # Write cartesian gradiend
    if len(gcc) != 0:
       str_gts += "# Non-scaled cartesian gradient [hartree/bohr]\n"
       str_gts += "start_grad\n"
       for at in range(nat):
           gxx,gyy,gzz = gcc[3*at:3*at+3]
           str_gts += "   %+15.8E  %+15.8E  %+15.8E\n"%(gxx,gyy,gzz)
       str_gts += "end_grad\n\n"
    # Write force constant matrix (i.e. hessian matrix)
    if len(Fcc) != 0:
       # matrix in triangular form
       if len(Fcc) == 3*nat: Fcc = matrix2lowt(Fcc)
       str_gts += "# Low triangular part of force constant (hessian) matrix [hartree/bohr^2]\n"
       str_gts += "# i.e.: F_11, F_21, F_22, F_13, F_23, F_33...\n"
       str_gts += "start_hess\n"
       for idx in range(0,len(Fcc),5):
           line = "  ".join(["%+15.8E"%Fij for Fij in Fcc[idx:idx+5]])
           str_gts += "   %s\n"%line
       str_gts += "end_hess\n\n"
    # Write freqs
    elif len(freqs) != 0:
       str_gts += "start_freqs\n"
       for idx in range(0,len(freqs),5):
           line = "  ".join(["%7.2f"%(freq) for freq in freqs[idx:idx+5]])
           str_gts += "           %s\n"%line
       str_gts += "end_freqs\n\n"
    # write
    with open(gtsfile,'w') as asdf: asdf.write(str_gts)
#===============================================#

#=======================================================#
# READING FUNCTION                                      #
#-------------------------------------------------------#
def read_molecule(filename):
    # read gaussian file
    data_gaulog = read_gaussian_log(filename)
    # split data
    ch      = data_gaulog[2]
    mtp     = data_gaulog[3]
    symbols = data_gaulog[4]
    xcc     = data_gaulog[5]
    gcc     = data_gaulog[6]
    Fcc     = data_gaulog[7]
    V0      = data_gaulog[8]
    level   = data_gaulog[11]
    # symbols to atomic numbers
    atonums = [atomic_numbers[s] for s in symbols]
    # return data
    return xcc, atonums, ch, mtp, V0, gcc, Fcc
#=======================================================#
def main():
   args = sys.argv[1:]

   # Get input file name
   print("")
   if len(args) == 0:
      fname = input("Insert name of file: ")
   else:
      fname = sys.argv[1]
      print("Filename: %s"%fname)

   # Print input file name
   if not os.path.exists(fname):
      print("   - file not found!\n")
   else: print("")

   # Read Gaussian file
   xcc, atonums, ch, mtp, V0, gcc, Fcc = read_molecule(fname)

   # Write gts file
   if fname.lower().endswith(".log"): gtsfile = fname[:-4]+".gts"
   else                             : gtsfile = fname     +".gts"
   print("File to be created:",gtsfile)
   write_gtsfile(xcc,atonums,ch,mtp,V0,"C1",1,gcc,Fcc,gtsfile)
   print(" - done!")
#=======================================================#


if __name__ == "__main__": main()
   

