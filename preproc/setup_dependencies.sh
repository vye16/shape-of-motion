# install additional dependencies for track-anything and depth-anything
pip install -r requirements_extra.txt

# install droid-slam
echo "Installing DROID-SLAM..."
cd DROID-SLAM
python setup.py install
cd ..

# install unidepth
echo "Installing UniDepth..."
cd UniDepth
pip install .
cd ..

# install tapnet
echo "Installing TAPNet..."
cd tapnet
pip install .
cd ..

echo "Downloading checkpoints..."
mkdir checkpoints
cd checkpoints
# sam_vit_h checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# xmem
wget -P ./saves/ https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth
# droid slam checkpoint
gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh
# tapir checkpoint
wget https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt
echo "Done downloading checkpoints"
