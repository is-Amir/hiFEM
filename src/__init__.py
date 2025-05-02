from src.hiFEM import *

# ---------------------------------------------------------------------------- #
#                          initializations and setups:                         #
# ---------------------------------------------------------------------------- #
# Notes: 
# - python 3.11 is necessary for pygalmesh to work
# - pyvista[all,trame] is necessary for the trame export to work
#
# conda create -n softsim python=3.11 jupyter scipy numpy -y
# pip install open3d lxml
# pip install pyvista[all,trame]
# pip install git+https://github.com/russelmann/confmap.git