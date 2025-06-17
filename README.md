# interpolation-mesh-graph-net
Interpolation Mesh-Graph Net using disconnected spatial graphs of higher order element based meshes - journal paper
by Sebestyen Jozsef Adam and Kim Do-Nyun, Seoul National University, 2025

Interpolation Mesh-Graph Net (IMGN) - the first GNN model that natively supports disconnected, HO element-based meshes without requiring full graph connectiviy. The model uses higher order element based meshes without mesh simplification or subdivision into first-order elements. IMGN offers a practical and scalable solution for physical simulations that depend on HO accuracy.

This repository contains the minimal code that runs the IMGN model on the Linear Elastostatic dataset.
The model includes the base MGN graph model and the multi-scale and interpolation processing modes of the IMGN model as well.
The training can be run with 1st to 5th order meshes and in the original high quality first order vesion as well.
