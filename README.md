# LineFitting
Tests of line fitting algorithms, with a focus on different strategies for multicomponent fitting. Description of key routines:

* `nh3_testcubes.py` -- Create a set of test cubes for running multicomponet fitting tests on (see "GAS Test Cubes" below for more details).

* `analysis_tools.py` -- Contains tools to analyze test results. A test result can be initialized as a `TestResults` object for quick analysis.

* `analysis_template.ipynb` -- A template to run identical analysis on different fitting methods. This notebook wraps around `analysis_tools.py` and uses `example_result.txt` as an example.

* `example_result.txt` -- The example test result to be used in tandem with `analysis_template.ipynb`.


## GAS Test Cubes
A test set of 10,000 cubes are generated via `nh3_testcubes.py` for a benchmark comparison between different line-fitting algorithms. A synthetic ammonia spectrum modelled by either one or two velocity slabs is generated at the central pixel of each test cube. The remainder pixels in the cube are filled with models nearly identical to that of the center pixel, differ only by a small velocity offset for individual velocity slabs.

Peformance of the line-fitting algorithms are compared against one another statically based on their fits to the central pixel of the cubes. Default random seed for the cube generation is set to `42` to accommodate further needs to make spectrum-to-spectrum comparisons.

### Distributions of the Test Set Parameters
The physical parameters used to generate the test cubes are randomly drawn from the following distributions for each velocity slap:
* Velocity centroid:
  - First slab: uniformly distributed in the range of `[-2.5, 2.5)` km s<sup>-1</sup>.
  - Second slab: uniformly distributed in the range of `[-2.5, 2.5)` km s<sup>-1</sup>, **with respect to** velocity centroid of the first slab.
* Linewidth:
  - Quadrature-sum of the thermal and the non-thermal linewidths
  - Thermal linewidth: `0.08` km s<sup>-1</sup>
  - Non-thermal linewidth: log-normally distributed in the natural log space. The distribution has an 1-sigma range of `[0.1, 1.6)` log(km s<sup>-1</sup>)
* Column Density:
  - Uniformly distributed in the log<sub>10</sub> space in the range of `[13, 14.5)` log(cm<sup>-2</sup>).
* Kinetic temperature:
  - Uniform distribution in the range of `[8, 25)` K

