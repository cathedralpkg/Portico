# Charge, multiplicity, energy [hartree],
# point group and rotational symmetry number
start_basic
   charge        0
   multiplicity  1
   energy       -113.18537710    # Total energy in hartree
   pointgroup    C1              # Point group
   rotsigma      1               # Rotational sigma
end_basic

# Atomic number and non-scaled cartesian coordinates [bohr]
start_cc
   006   +1.23431241E-01  +9.87275601E-01  +0.00000000E+00
   008   +1.23431241E-01  -1.14460475E+00  +0.00000000E+00
end_cc

# Non-scaled cartesian gradient [hartree/bohr]
start_grad
   +0.00000000E+00  +0.00000000E+00  +0.00000000E+00
   +0.00000000E+00  +0.00000000E+00  +0.00000000E+00
end_grad

# Low triangular part of force constant (hessian) matrix [hartree/bohr^2]
# i.e.: F_11, F_21, F_22, F_13, F_23, F_33...
start_hess
   +3.49720000E-04  +0.00000000E+00  +1.34804276E+00  +0.00000000E+00  +0.00000000E+00
   +3.49720000E-04  -3.49720000E-04  +0.00000000E+00  +0.00000000E+00  +3.49720000E-04
   +0.00000000E+00  -1.34804276E+00  +0.00000000E+00  +0.00000000E+00  +1.34804276E+00
   +0.00000000E+00  +0.00000000E+00  -3.49720000E-04  +0.00000000E+00  +0.00000000E+00
   +3.49720000E-04
end_hess

