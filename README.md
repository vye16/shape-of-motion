# Shape of Motion: 4D Reconstruction from a Single Video
**[Project Page](TODO) | [Arxiv](TODO)**

[Qianqian Wang](https://qianqianwang68.github.io/)<sup>1,2</sup>*, [Vickie Ye](https://people.eecs.berkeley.edu/~vye/)<sup>1</sup>\*, [Hang Gao](https://hangg7.com/)<sup>1</sup>\*, [Jake Austin](https://www.linkedin.com/in/jakeaustin4701)<sup>1</sup>, [Zhengqi Li](https://zhengqili.github.io/)<sup>2</sup>, [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/)<sup>1</sup>

<sup>1</sup>UC Berkeley   &nbsp;  <sup>2</sup>Google Research

\* Equal Contribution



## Installation

```
conda create -n som python=3.10
conda activate som
```

Update `requirements.txt` with correct CUDA version for PyTorch and cuUML, 
i.e., replacing `cu122` and `cu12` with your CUDA version. 
```
pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

## Usage

### Preprocessing

We depend on the third-party libraries in `preproc` to generate depth maps, object masks, camera estimates, and 2D tracks.
Please follow the guide in the [preprocessing README](./preproc/README.md).

### Fitting to a Video

```
python run_training.py --work-dir <OUTPUT_DIR> --data:davis --data.seq-name horsejump-low
```

## Evaluation on iPhone Dataset
First, download our processed iPhone dataset from [this](https://drive.google.com/drive/folders/1xJaFS_3027crk7u36cue7BseAX80abRe?usp=sharing) link. To train on a sequence, e.g., *paper-windmill*, run:

```
python run_training.py --work-dir <OUTPUT_DIR> --port <PORT> --data:iphone --data.data-dir </path/to/paper-windmill/>
```

After optimization, the numerical result can be evaluated via:
```
PYTHONPATH='.' python scripts/evaluate_iphone.py --data_dir </path/to/paper-windmill/> --result_dir <OUTPUT_DIR> --seq_names paper-windmill  
```


## Citation
```
@inproceedings{som2024,
  title     = {Shape of Motion: 4D Reconstruction from a Single Video},
  author    = {Wang, Qianqian and Ye, Vickie and Gao, Hang and Austin, Jake and Li, Zhengqi and Kanazawa, Angjoo},
  journal   = {arXiv preprint arXiv:2407.13764},
  year      = {2024}
}
```
