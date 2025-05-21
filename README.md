<p align="center">
  <h1 align="center">MonoSplat: Generalizable 3D Gaussian Splatting from Monocular Depth Foundation Models</h1>


## Installation

To get started, clone this project, create a conda virtual environment using Python 3.10+, and install the requirements:

```bash
git clone https://github.com/CUHK-AIM-Group/MonoSplat.git
cd MonoSplat
conda create -n monosplat python=3.10
conda activate monosplat
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Acquiring Datasets

### RealEstate10K and ACID

Our MVSplat uses the same training datasets as pixelSplat and MVSplat. Below we quote pixelSplat's [detailed instructions](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) on getting datasets.

### DTU (For Testing Only)

* Download the preprocessed DTU data [dtu_training.rar](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view).
* Convert DTU to chunks by running `python src/scripts/convert_dtu.py --input_dir PATH_TO_DTU --output_dir datasets/dtu`

## Running the Code

### Evaluation

To render novel views and compute evaluation metrics from a pretrained model,

* get the [pretrained models](), and save them to `/checkpoints`

* run the following:

```bash
# re10k
python -m src.main +experiment=re10k \
    mode=test \
    dataset/view_sampler=evaluation \
    checkpointing.load=/path/to/checkpoint \
    dataset.view_sampler.index_path=assets/evaluation_index_re10k_nctx10.json \
    test.compute_scores=true \
    wandb.mode=disabled
```

* the rendered novel views will be stored under `outputs/test`

### Training

Run the following:

```bash
# download the backbone pretrained weight from unimatch and save to 'checkpoints/'
wget 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth' -P checkpoints
# train mvsplat
python -m src.main +experiment=re10k data_loader.train.batch_size=14
```

Our models are trained with a single A100 (80GB) GPU. They can also be trained on multiple GPUs with smaller RAM by setting a smaller `data_loader.train.batch_size` per GPU.

### Cross-Dataset Generalization

We use the default model trained on RealEstate10K to conduct cross-dataset evaluations. To evaluate them, *e.g.*, on DTU, run the following command

```bash
# RealEstate10K -> DTU
python -m src.main +experiment=dtu \
    mode=test \
    checkpointing.load=/path/to/checkpoint \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.index_path=assets/evaluation_index_dtu_nctx2.json \
    test.compute_scores=true \
    wandb.mode=disabled
```

## BibTeX

```bibtex
```

## Acknowledgements

The project is largely based on [pixelSplat](https://github.com/dcharatan/pixelsplat) and [MVSplat](https://github.com/dcharatan/pixelsplat](https://github.com/donydchen/mvsplat ). Many thanks to these two projects for their excellent contributions!
