# Charge, multiplicity, energy [hartree],
# point group and rotational symmetry number
start_basic
   charge        0
   multiplicity  1
   energy       -188.38056650    # Total energy in hartree
   pointgroup    C1              # Point group
   rotsigma      1               # Rotational sigma
end_basic

# Atomic number and non-scaled cartesian coordinates [bohr]
start_cc
   006   +0.00000000E+00  +0.00000000E+00  +0.00000000E+00
   008   -2.18935243E+00  +0.00000000E+00  +0.00000000E+00
   008   +2.18935243E+00  +0.00000000E+00  +0.00000000E+00
end_cc

# Non-scaled cartesian gradient [hartree/bohr]
start_grad
   +0.00000000E+00  +0.00000000E+00  +0.00000000E+00
   +1.26000000E-06  +0.00000000E+00  +0.00000000E+00
   -1.26000000E-06  +0.00000000E+00  +0.00000000E+00
end_grad

# Low triangular part of force constant (hessian) matrix [hartree/bohr^2]
# i.e.: F_11, F_21, F_22, F_13, F_23, F_33...
start_hess
   +2.09423881E+00  +0.00000000E+00  +1.47610600E-01  +0.00000000E+00  +0.00000000E+00
   +1.47610600E-01  -1.04711941E+00  +0.00000000E+00  +0.00000000E+00  +1.13331373E+00
   +0.00000000E+00  -7.38053000E-02  +0.00000000E+00  +0.00000000E+00  +3.69249700E-02
   +0.00000000E+00  +0.00000000E+00  -7.38053000E-02  +0.00000000E+00  +0.00000000E+00
   +3.69249700E-02  -1.04711941E+00  +0.00000000E+00  +0.00000000E+00  -8.61943300E-02
   +0.00000000E+00  +0.00000000E+00  +1.13331373E+00  +0.00000000E+00  -7.38053000E-02
   +0.00000000E+00  +0.00000000E+00  +3.68803300E-02  +0.00000000E+00  +0.00000000E+00
   +3.69249700E-02  +0.00000000E+00  +0.00000000E+00  -7.38053000E-02  +0.00000000E+00
   +0.00000000E+00  +3.68803300E-02  +0.00000000E+00  +0.00000000E+00  +3.69249700E-02
end_hess

