
We depend on the following third-party libraries for preprocessing:

1. Metric depth: [Unidepth](https://github.com/lpiccinelli-eth/UniDepth/blob/main/install.sh)
2. Monocular depth: [Depth Anything](https://github.com/LiheYoung/Depth-Anything)
3. Mask estimation: [Track-Anything](https://github.com/gaomingqi/Track-Anything) (Segment-Anything + XMem)
4. Camera estimation: [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM/tree/main)
5. 2D Tracks: [TAPIR](https://github.com/google-deepmind/tapnet)

We provide a setup script in `setup_dependencies.sh` for setting up the environments for running these.
```
./setup_dependencies.sh
```
