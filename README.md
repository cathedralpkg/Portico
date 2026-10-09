# PORTICO

**P**rojection **O**nto **R**oto-**T**ranslational and **I**nternal **CO**ordinates

```text
+------------------------------------+
|                   _   _            |
|  _ __   ___  _ __| |_(_) ___ ___   |
| | '_ \ / _ \| '__| __| |/ __/ _ \  |
| | |_) | (_) | |  | |_| | (_| (_) | |
| | .__/ \___/|_|   \__|_|\___\___/  |
| |_|                                |
+------------------------------------+
```

PORTICO is a Python program for the **asymptotic classification of
transition-state normal modes** in molecular dissociation and fragmentation.

For a dissociation channel

```text
R  →  TS‡  →  P1 + P2
```

some degrees of freedom that are vibrational at the transition state
correspond asymptotically to rotations or translations of the product
fragments. PORTICO automatically identifies the normal modes assigned to
this roto-translational sector, without manual inspection of normal-mode
animations and without propagating the reaction path toward the asymptotic
product region.

The current implementation is restricted to channels yielding **two product
fragments**, including atomic, linear, and nonlinear products.

## How it works

PORTICO constructs a mixed representation at the supplied dividing-surface
geometry, which may be stationary or non-stationary,

```text
B = [ Bv
      Bt,r ]
```

from two blocks:

- **Bv** — rows obtained by evaluating, at the dividing-surface geometry,
  a complete, non-redundant set of internal coordinates selected to span the
  vibrational degrees of freedom of the isolated product fragments;

- **Bt,r** — normalized rigid-body translational and rotational displacement
  vectors of the product fragments constructed at the same dividing-surface
  geometry.

The roto-translational rows are constructed geometrically from the fragment
centroids and the principal axes of their geometric inertia tensors. With
this construction, the rows of **Bv** are orthogonal to those of **Bt,r** in
the Euclidean metric used for the classification, while the rows within
**Bt,r** are mutually orthonormal.

The normal modes are expressed in this mixed representation through the
Wilson GF formalism. For each mode *i*, PORTICO computes a scalar
roto-translational weight,

**Ω<sub>i</sub> ∈ [0, 1]**,

from the components associated with **Bt,r**.

The number of normal modes assigned to the roto-translational sector is fixed
by the dimensionality of the dissociation channel. For a stationary
first-order saddle point, the imaginary-frequency mode is assigned to this
sector by construction because it defines the reaction coordinate. The
remaining required roto-translational modes are selected from the
real-frequency modes by ranking their Ω<sub>i</sub> values. No empirical
threshold in Ω<sub>i</sub> is used.

PORTICO can also analyze **non-stationary geometries**. With

```text
stationary no
```

the gradient direction is projected out and plays the role of the reaction
coordinate. This allows analysis of points along a reaction path or externally
defined non-stationary dividing surfaces.

For nominally linear products that are appreciably bent at the
supplied dividing-surface geometry, PORTICO replaces the pair of linear-bending
coordinates by the corresponding angular-bending coordinate evaluated at that
geometry and the rigid-rotation row associated with the smallest principal
moment of inertia. This preserves the separation between the
product-vibrational and roto-translational blocks.

## Requirements

Core PORTICO requires:

- Python ≥ 3.8
- `numpy`
- `scipy`
- `matplotlib`

The package also installs the `gaussian2gts` converter. If that utility has
additional dependencies in your installation, consult its package metadata or
script header.

## Installation

### From PyPI

```bash
pip install cathedralpkg-portico
```

### From GitHub

```bash
pip install git+https://github.com/cathedralpkg/portico.git
```

### In a conda environment

```bash
conda create -n portico python=3.11
conda activate portico
pip install cathedralpkg-portico
```

Any of the above installs the `portico` and `gaussian2gts` commands in your
`PATH`.

The scripts can also be executed directly with Python if the required
dependencies are available.

## Usage

```bash
portico input_file_name          # run a classification
portico -h | --help              # show help
portico -g | --ginput            # generate an example input file
portico -v | --version           # show program version
```

An example input file can be generated with:

```bash
portico --ginput
```

This creates `portico.inp` in the current directory, unless a file with that
name already exists.

## Input file

A typical input file is:

```text
# Files with the electronic-structure data (gts format)
file_divsurf  DS.gts        # dividing-surface structure (stationary or non-stationary)
file_product1 P1.gts        # product fragment 1
file_product2 P2.gts        # product fragment 2

# Atom mapping (1-based): each atom of a product is assigned to one
# atom of the dividing-surface structure; every atom must appear exactly once
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
eps_ccic        6.00          # max. |freq(cc) - freq(ic)| (cm^-1)
linear_cutoff   170.0         # degrees; bond angles >= this value are treated as linear
stationary      yes           # stationary point? yes/no (no: non-stationary dividing surface)
#seed           9876543210    # random seed; default: taken from the time
#                             # (printed in the output); fix it to repeat a run
#-------------------------------------------------------------------#
```

### Required keywords

| Keyword | Meaning |
|---|---|
| `file_divsurf` | `.gts` file for the dividing-surface structure to be analyzed |
| `file_product1` | `.gts` file for product fragment 1 |
| `file_product2` | `.gts` file for product fragment 2 |

### Atom mapping

The atom mapping is **1-based**.

Every atom of each product must be mapped, every atom of the dividing-surface
structure must appear exactly once across the two product mappings, and mapped
atoms must correspond to the same chemical element.

PORTICO checks the mapping before the classification starts and aborts if the
mapping is incomplete, duplicated, out of range, or chemically inconsistent.

### Optional keywords

| Keyword | Default | Meaning |
|---|---:|---|
| `file_plot` | `omegas.png` | output file for the Ω<sub>i</sub> bar plot |
| `eps_conn` | `1.30` | scale factor used to infer molecular connectivity |
| `eps_ccic` | `6.00` | maximum allowed difference between Cartesian- and internal-coordinate frequencies, in cm⁻¹ |
| `linear_cutoff` | `170.0` | bond-angle threshold in degrees; angles at or above this value are treated as linear |
| `stationary` | `yes` | `yes` for a stationary dividing surface; `no` for a non-stationary dividing surface |
| `seed` | current time | random seed used when selecting a non-redundant product-vibrational coordinate set |

If `seed` is omitted, PORTICO initializes the random seed from the current
time and prints it in the output. Specify it explicitly to reproduce the same
product-vibrational coordinate selection.

## Required electronic-structure data

PORTICO reads electronic-structure data from plain-text `.gts` files.

For the dividing-surface structure and for each isolated product, the file
contains:

- Cartesian geometry;
- Cartesian energy gradient;
- Cartesian Hessian;
- charge;
- multiplicity;
- electronic energy;
- point-group information;
- rotational symmetry number.

For a stationary point the gradient is numerically zero. For

```text
stationary no
```

a non-zero gradient is required and its direction is projected out before the
mode classification.

## Output

PORTICO reports the roto-translational weight Ω<sub>i</sub> of the analyzed
normal modes and identifies the modes assigned to the roto-translational
sector.

For a stationary first-order saddle point, the imaginary-frequency mode is
assigned to this sector by construction and is reported as `ifreq`.

Example:

```text
      freq (cm^-1)   Omega_i
    --------------------------
        -1531.2        ifreq   [roto-translational]
         1266.7        0.955   [roto-translational]
         2210.0        0.954   [roto-translational]
          812.1        0.934   [roto-translational]
          715.0        0.907   [roto-translational]
         1644.3        0.706
         1166.2        0.537
         ...
```

The output also includes diagnostic information such as:

- the number of product-vibrational, translational, and rotational rows;
- the shape and rank of the mixed **B** matrix;
- whether the mixed matrix is full rank;
- numerical checks of the orthogonality between the different row blocks;
- comparison of Cartesian and mixed-representation frequencies;
- the random seed used for the calculation;
- ΔΩ, when available, between the last selected roto-translational mode and
  the next mode in the Ω<sub>i</sub> ranking.

The Ω<sub>i</sub> plot is written to the file specified by `file_plot`.

## The `.gts` file format

The `.gts` format is a plain-text format shared by programs in the
**Cathedral** package. It stores the electronic-structure data of a single
molecular structure.

A `.gts` file is organized in blocks delimited by `start_<block>` and
`end_<block>` keywords. Lines beginning with `#` are comments and are
ignored. Quantities used by PORTICO are expressed in atomic units.

### `start_basic` / `end_basic`

Scalar properties of the structure:

| Keyword | Meaning |
|---|---|
| `charge` | total charge |
| `multiplicity` | spin multiplicity |
| `energy` | total electronic energy, in hartree |
| `pointgroup` | point group, e.g. `C1`, `C2v` |
| `rotsigma` | rotational symmetry number |

### `start_cc` / `end_cc`

One line per atom: atomic number followed by the three Cartesian coordinates
in bohr.

Example:

```text
006   x   y   z
```

### `start_grad` / `end_grad`

Cartesian energy gradient in hartree/bohr, one atom per line with three
components.

For a stationary point the gradient is numerically zero. A non-zero gradient
is required for the analysis of a non-stationary geometry.

### `start_hess` / `end_hess`

The Cartesian Hessian is stored in lower-triangular form, in
hartree/bohr²:

```text
F_11, F_21, F_22, F_31, F_32, F_33, ...
```

The values are written sequentially; the program reconstructs the full
symmetric `3N × 3N` matrix.

## Converting Gaussian outputs to `.gts`

The helper utility `gaussian2gts` converts a Gaussian log file from a
frequency calculation into `.gts` format:

```bash
gaussian2gts DS.log
gaussian2gts P1.log
gaussian2gts P2.log
```

If another electronic-structure package is used, an analogous converter can
be written to produce the same plain-text `.gts` format.

## Notes on the mixed representation

The roto-translational rows are geometric rigid-body displacement vectors,
not scalar internal coordinates. In particular, the rotational rows are
constructed without atom-dependent mass weighting.

The product-vibrational rows are orthogonal to the roto-translational rows in
the Euclidean metric used for the classification. The rows within the
product-vibrational block are not generally mutually orthogonal, so the
numerical value of Ω<sub>i</sub> depends on the selected non-redundant
internal-coordinate representation. PORTICO therefore uses the **ordering**
of the Ω<sub>i</sub> values, together with the channel dimensionality, rather
than an empirical absolute threshold.

At non-stationary geometries, the gradient-dependent curvature terms are
included for rows derived from scalar internal coordinates. Rotational
displacement rows do not correspond to gradients of scalar coordinates; the
associated second-derivative contribution is therefore omitted. This
approximation is discussed in the accompanying publication.

## Current scope

The current implementation supports dissociation channels yielding two
fragments and can treat atomic, linear, and nonlinear products.

PORTICO does not locate a transition state, variational dividing surface, or
free-energy bottleneck. It analyzes a supplied stationary or non-stationary
dividing-surface geometry and classifies its local modes.

## Citation

If you use PORTICO in your work, please cite:

> D. Ferro-Costas, *Portico: a program for the asymptotic classification of
> transition-state normal modes*, submitted to *Computer Physics
> Communications* (2026).

Please replace the provisional citation above with the final bibliographic
information once the article is published.

## License

Distributed under the MIT License. See `LICENSE` for details.

## Author

David Ferro-Costas — Universidade de Santiago de Compostela  
ORCID: 0000-0002-8365-4047
