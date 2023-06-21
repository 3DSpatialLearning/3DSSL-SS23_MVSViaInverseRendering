# 3DSSL-SS23_MVSViaInverseRendering

This is our project for the practical course 3D Scanning & Spatial Learning.

## Goal

The goal was updated.

Primary focus of this project is get realistic geometry out of different models. sdfstudio already achieves reasonable
results.

The goal, as formulated in the final project plan, is to extend sdfstudio to support relighting and separate
material properties into different MLPs. For this, the [sdfstudio](https://github.com/autonomousvision/sdfstudio) project
will be extended.

## Project structure

```
├── notebooks                - Contains experimental code to showcase ideas/visualization on remote machine
├── scripts                  - Every script external and not needed for extending sdfstudio
│       ├── preprocess              - Convert dataset to various formats (nerfstudio/sdfstudio/own) for training
│       ├── training                - Training pipeline
│       └── visualization           - Visualizing results from training (pyvista/dreifus)
└── sdfstudio                - Modified sdfstudio base to support our project
```

# Project Layout

All project related files are stored on the provided remote machine. The data used as input, as well as the models
and extracted geometry is stored according to the following convention.

## Datasets

All data resides on `/mnt/hdd/data` on the remote machine. The raw data, as well as the preprocessed datasets, are stored.
When generating new dataset, please specify rough format and processing in the following.


Currently, the following datasets are:
```
18.tar.xz   - Raw data, head_18
38.tar.xz   - Raw data, head_38
85.tar.xz   - Raw data, head_85
97.tar.xz   - Raw data, head_97
124.tar.xz  - Raw data, head 124
240.tar.xz  - Raw data, head_240

mesh_85_f_150_prior     - Ground truth mesh (laser scan) of head_85 in frame 150

scripts         - Folder containg python scripts for processing; OBSOLETE/OUTDATED; DONT USE

experimenting   - Dataset of unspecified format; Can be used to try different settings
                - Exact format specified in 'About.md' file in folder

head_01     - 16 RGB images and estimated (not accurate) binary masks
            - Those masks were estimated with rembg and can be used for guiding the ray sampling in sdfstudio
            - File and mask names represent the same capture
            -  Image shape (width,height) = (2200,3208)
            
(INCOMPLETE)
head_18     - 16 RGB images with metadata (normals/depth/consistency)
            - Indexing WRONG; NO extrinsics/intrinsics
            - Image shape (width,height) = (2200,3208)
            
head_18_d_2 - Downsampled version (factor = 2) of the head_18 dataset
            - Image shape: (width,height,channels) = (1100, 1604,3)
            - Downsampling was performed with script scripts/preprocess/custom_to_sdfstudio.py
            - Contains additionally normal/depth maps as PNG files for inspecting their validity
            - Image shape: (width, height) = (1100, 1604)
            
head_38     - 16 RGB images with metadata (normals/depth/consistency)
            - Two frames (0/151); extrinsics/intrinsics availabel
            - Image shape (width,height) = (2200,3200)
        
head_85     - 16 RGB images with metadata (normals/depth/consistency)
            - Two frames (0/120); extrinsics/intrinsics availabel
            - Image shape: (width, height) = (2200, 3208)
            
(INCOMPLETE)
head_97         - 16 RGB images with metadata
                - Indexing WRONG; NO extrinsics/intrinsics
            
head_124        - 16 RGB images with metadata (one frame)

head_240        - 16 RGB images with metadata (one frame

head_240_d_2    - Same as head_240; Downsampled by factor of 2
                - Processed with custom_to_sdfstudio.py
```

## Models
Trained models are stored in `/mnt/hdd/models`.

## Output
Output in the form of meshes,textures,etc. is stored in `/mnt/hdd/output`.


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