# install additional dependencies for track-anything and depth-anything
pip install -r requirements_extra.txt

# install droid-slam
cd DROID-SLAM
pip install .
cd ..

# install unidepth
cd UniDepth
pip install .
cd ..

# install tapnet
cd tapnet
pip install .
cd ..
