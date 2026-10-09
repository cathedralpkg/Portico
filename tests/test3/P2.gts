# Charge, multiplicity, energy [hartree],
# point group and rotational symmetry number
start_basic
   charge        0
   multiplicity  1
   energy       -1.17175460      # Total energy in hartree
   pointgroup    C1              # Point group
   rotsigma      1               # Rotational sigma
end_basic

# Atomic number and non-scaled cartesian coordinates [bohr]
start_cc
   001   -1.63589309E+00  +2.16449247E+00  +3.13159581E-01
   001   -2.07755841E-01  +2.07651216E+00  +2.53395429E-01
end_cc

# Non-scaled cartesian gradient [hartree/bohr]
start_grad
   +0.00000000E+00  +0.00000000E+00  +0.00000000E+00
   +0.00000000E+00  +0.00000000E+00  +0.00000000E+00
end_grad

# Low triangular part of force constant (hessian) matrix [hartree/bohr^2]
# i.e.: F_11, F_21, F_22, F_13, F_23, F_33...
start_hess
   +3.68898250E-01  -2.27271800E-02  +1.38027000E-03  -1.54383500E-02  +9.51080000E-04
   +6.26220000E-04  -3.68898250E-01  +2.27271800E-02  +1.54383500E-02  +3.68898250E-01
   +2.27271800E-02  -1.38027000E-03  -9.51080000E-04  -2.27271800E-02  +1.38027000E-03
   +1.54383500E-02  -9.51080000E-04  -6.26220000E-04  -1.54383500E-02  +9.51080000E-04
   +6.26220000E-04
end_hess

