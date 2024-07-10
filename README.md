# Shape of Motion
[Qianqian Wang*](https://qianqianwang68.github.io/) [Vickie Ye*](https://people.eecs.berkeley.edu/~vye/) [Hang Gao*](https://hangg7.com/) ...

## Installation

```
pip install -e .
```

## Usage

### Preprocessing
We depend on the third-party libraries in `preproc` to generate monocular depth maps, camera estimates, and long-range 2D tracks. 

### Fitting to a Video

```
python run_training.py --work-dir <OUTPUT> --data:davis --data.seq-name horsejump-low
```

## Evaluation

### iPhone Dataset
