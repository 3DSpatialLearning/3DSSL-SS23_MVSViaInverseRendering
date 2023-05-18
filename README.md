# 3DSSL-SS23_MVSViaInverseRendering

This is our project for the practical course 3D Scanning & Spatial Learning.

## Goal

The goal, as formulated in the final project plan, is to extend sdfstudio to support relighting and separate
material properties into different MLPs. For this, the !(https://github.com/autonomousvision/sdfstudio)[sdfstudio] project
will be extended.

## Structure

```
├── notebooks   - Contains experimental code to showcase idea
├── scripts     - Every script external and not needed for extending sdfstudio
└── sdfstudio   - Modified sdfstudio base to support our project
```

## Setup
### Develop
In case you cannot run python scripts because your conda environment does not find the `nerfstudio` module, add a file
`conda.pth` to `~/miniconda3/envs/<ENV_NAME>/lib/pythonX.X/site-packages` with a path to the `sdfstudio` project.

### Build
Create a new conda environment using the `environment.yml` file:

```
conda env create -f environment.yml
```

Activate the environment

```
conda activate 3dssl_env
```

## Jupyter notebook

In order to use the conda environment with a Jupyter notebook, install 

```
conda install ipykernel
```

## nerfstudio

To use the nerfstudio viewer, you need ssh with port forwarding (nerfstudio is using port 7007 by default):

```
ssh -L 7007:localhost:7007 <username>@<hostname>
```

I copied the toy model for nerfstudio over to the HDD, so you can run webviewer with:

```
ns-viewer --load-config /mnt/hdd/models/nerfstudio/outputs/poster/nerfacto/2023-04-19_222320/config.yml
```

Now you should be able to access it with your browser.


## colmap

Colmap is installed globally. I copied over the images to `/mnt/hdd/data/msv_images`. To test the dense matching and reconstruction use

```
colmap automatic_reconstructor --use_gpu=1 --dense=1 --workspace_path <YOUR_PATH> --image_path /mnt/hdd/data/mvs_images/
```

To copy over the mesh to your machine you can use `scp`:

```
scp <username>@<hostname>:<path-to-mesh-file>(dense/0/fused.ply) fused.ply
```

## Papers

## Relevant Links

SDFstudio in nerfstudio documentation: https://docs.nerf.studio/en/latest/extensions/sdfstudio.html <br />
Extracting Triangular 3D Models, Materials, and Lighting From Images: https://nvlabs.github.io/nvdiffrec/ <br />
NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction: https://arxiv.org/abs/2106.10689 <br />
High-Quality Single-Shot Capture of Facial Geometry: https://studios.disneyresearch.com/wp-content/uploads/2019/03/High-Quality-Single-Shot-Capture-of-Facial-Geometry.pdf
