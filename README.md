Plan for hybrid MPI/OPENMP Project 
Eduardo, Alitzel, Rainette 

Tasks (what they mentioned yesterday):
Experiment design
Report
Create bash scripts
Code baseline (grid constant, rectangular domain)
Profiling
Extension code (eg: different domain shapes)
Research questions to answer:
Find optimum MPI/OpenMP division
Try for different domain sizes nx = ny
Different domain shape nx != ny
Changing numbers of ranks per x/y dimension

5) Best configuration of a hybrid (OpenMP / MPI) weather and climate model: Typical weather and climate models use OpenMP to parallelize in the vertical and MPI to domain-decompose in the horizontal. Implement domain decomposition using MPI in the stencil2d-kparallel.F90 version of the code. Find the optimal configuration of running the code on an Alps node (4 x Grace-Hopper). How many threads and how many MPI ranks give the best performance? Can you understand why?

Before mid-july: implement partitioner in fortran and finish project ideas

Rainette not available: 7-14 July
Ali not available: 8-22 July, 26 July - 10 August, 13 August


Meet in zurich: somewhere week of 11-22 Aug
