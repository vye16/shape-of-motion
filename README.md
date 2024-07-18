# Shape of Motion

[Qianqian Wang\*](https://qianqianwang68.github.io/) [Vickie Ye\*](https://people.eecs.berkeley.edu/~vye/) [Hang Gao\*](https://hangg7.com/) ...

## Installation

```
pip install -e .
pip install --upgrade \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==24.6.*" \
    "dask-cudf-cu12==24.6.*" \
    "cuml-cu12==24.6.*" \
    "cugraph-cu12==24.6.*" \
    "cuspatial-cu12==24.6.*" \
    "cuproj-cu12==24.6.*" \
    "cuxfilter-cu12==24.6.*" \
    "cucim-cu12==24.6.*" \
    "pylibraft-cu12==24.6.*" \
    "raft-dask-cu12==24.6.*" \
    "cuvs-cu12==24.6.*"
```

## Development

```
cd .git
ln -sf ../../.dev/pre-commit
```

## Usage

### Preprocessing

We depend on the third-party libraries in `preproc` to generate monocular depth maps, camera estimates, and long-range 2D tracks.

### Fitting to a Video

```
python run_training.py --work-dir <OUTPUT_DIR> --data:davis --data.seq-name horsejump-low
```

## Evaluation on iPhone Dataset
First, download our processed iPhone dataset from [this](todo) link. To train on a sequence, e.g., *paper-windmill*, run:

```
python run_training.py --work-dir <OUTPUT_DIR> --port <PORT> --data:iphone --data.data-dir </path/to/paper-windmill/>
```

After optimization, the numerical result can be evaluated via:
```
PYTHONPATH='.' python scripts/evaluate_iphone.py --data_dir </path/to/paper-windmill/> --result_dir <OUTPUT_DIR> --seq_names paper-windmill  
```