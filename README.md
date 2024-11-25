# Chebychev Parameterization for Woven Fabric Modeling

This is the implementation of the paper "Chebychev Parameterization for Woven Fabric Modeling" by Annika Oehri, Aviv Segall, Jing Ren and Olga Sorkine-Hornung from SIGGRAPH ASIA 2024. 

**BibTex**
```
@article{Oehri:ChebyWoven:2024,
author = {Oehri, Annika and Segall, Aviv and Ren, Jing and Sorkine-Hornung, Olga},
title = {Chebyshev Parameterization for Woven Fabric Modeling},
journal = {ACM Transactions on Graphics},
volume = {43},
number = {6},
note = {SIGGRAPH ASIA 2024 issue},
year = {2024},
url = {https://doi.org/10.1145/3687928},
doi = {10.1145/3687928},
}
```

## Installation guide and Overview

The code is written in C++ and only depends on [libigl](https://github.com/libigl/libigl) (v2.5.0), for more explanations on the compilation of libigl projects please refer to [this](https://libigl.github.io/). With an existing libigl folder, you should be able to compile and run the project via the standard CMake-routine:
```
mkdir build
cd build
cmake ..
make
```
The project contains two parts, one being the code for the Chebyshev Parameterization method and the other an interactive Chebyshev deformation framework. For both, a few data examples from the paper are supplied in the data folder.


## Parameterization
The relevant code for this is in `cheby_parameterization/src/main.cpp`. The overall objective of the code is to provide a Chebyshev net parameterization for an input disk-topology mesh. 

**Input:** The mesh to parameterize. This mesh must be disk-topology. 

**How to use:**
* Start the application with the path to the mesh as the first command line argument
* Optionally, select constraints (anchor, warp and weft direction) for the parameterization. If these are not selected, two points are automatically selected to be fixed for LSCM and one for Chebyshev and ARAP parameterization.
* Press `3` to initialize the parameterization via LSCM
* Press `6`to compute the Chebyshev parameterization. This will automatically run until convergence/ the iteration limit is reached. To adapt these, adapt convergence_precision and max_iters in the code. If you want the shearing to be limited, set the shearing angle limit in the GUI beforehand.
* Press `4`to run ARAP parameterization. Again, this automatically runs until convergence.
* You can adapt the Texture Resolution by changing the number in the GUI
* To run from a saved initialization, you can use the button `load texture from .obj` and then run Chebyshev parameterization as per usual.
* To save your results with corresponding metrics like Chebyshev error, runtime etc, enter a name and click on `save .obj and stats`. It will write into the `res` folder.


## Deformation
The relevant code for this is in `cheby_deformation/src/main.cpp`. The overall objective of the code is to provide a framework for interactively deforming a (given) Chebyshev net. 

**Input:** The mesh to deform (*Chebyshev net*), optionally a collision mesh for draping. The Chebyshev net should be disk-topology or have a precomputed disjoint parameterization. The collision mesh can in theory be any mesh, however it is strongly advised to use a low-resolution mesh as the time needed to compute mesh intersections will impact the interactivity of the application.

**How to use:**
* Start the application with the path to the Chebyshev net as the first command line argument, and the path to the collision mesh as an optional second command line argument
* Provide the Chebyshev net in rest state by either of the following two options:
  * Press `1` to compute the LSCM parameterization of the input mesh to initialize and press `2` until convergence to get the Chebyshev parameterization. Note that this can only be done for disk topology meshes. For more complex parameterizations, including constraints and alike, please use the main parameterization code and import the texture to this application as explained next.
  * Use the `load texture from .obj` button to load a precomputed saved texture from the .obj input file. This also allows manipulating e.g. dresses consisting of multiple panels. For sewn together clothes, we assume the texture coordinates of different panels are fully disjoint and seams being represented by multiple texture coordinates mapping to the same mesh coordinate.
* Note that you can change the texture resolution in the GUI by entering a number if the scale does not visualize it well
* Create handles and move them around!
  * Press `p` or select `VERTEX` in the Handle Option dropdown menu to create a vertex handle and click on the wanted vertex.
  * Press `m` or select `MARQUE` in the Handle Option dropdown menu to create an area handle (rectangular, press `m` again for circular tool) and drag your mouse over the wanted area
  * Press `l` or select `LASSO` in the Handle Option dropdown menu to draw a freehand area selection (or polygonal if you press `l` again)
  * Press `x` or select `REMOVE` in the Handle Option dropdown menu and click on a handle in order to remove that constraint
  * There are three ways of manipulating the handles available, the standard translation invoked by pressing `t` or selecting `TRANSLATE` in the transformation mode dropdown menu, rotation by pressing `r` or selecting `ROTATE`, or scaling by selecting `SCALE`
  * Switch between handles by clicking on them (if necessary press `p` beforehand)
  * Move them around and watch the Chebyshev net change, if you are happy with your constraints but want to keep solving the system for convergence, press `c` to compute a few iterations with the same constraints
* To regularize shearing, enter a number ([0,1)) for ARAP regularization or specify a shearing angle limit for the projection step
* To regularize bending, enter a number ([0,1)) for bending regularization 
* To use gravity and dynamics, check the gravity checkbox and adapt the parameters until you're happy with the results
* To use collision, check the collision checkbox. Note that this only works when gravity is on. The collision works by incrementally pushing out vertices that enter the collision mesh, so not all intermediate results are correct, but in practice it converged well, and this best works when not making too drastic changes, e.g. move the handles smoothly along.
* If you like an intermediate result, you can press `save .obj` to save the mesh (with its parameterization) for further use :)
