### Model 001
#### Vanilla colmap for subject 85 frame 150
did not work


### Model 002
#### Vanilla colmap with shared intrinsics
did not work

### Model 003
#### Vanilla colmap without masks or shared intrinsics


### Model 003
#### Org extrinsics with colmap prior
"base_mesh": "data/3dssl/frame_00150_rescaled/mesh/mesh.obj"
"base_mesh": "out/3dssl/subject85.150-10-colmap-w-mesh/mesh/mesh.obj",

    "base_mesh": "out/3dssl/subject85.150-10-colmap/mesh/mesh.obj",
    "base_mesh": "out/3dssl/subject85.150-21-colmap/dmtet_mesh/mesh.obj",

### Model 24

```json
    {"ref_mesh": "data/3dssl/s850_rescaled",
    "base_mesh": "out/3dssl/subject85.150-23-colmap-w-mesh/mesh/mesh.obj",
    "random_textures": true,
    "sdf_regularizer": 0.01,
    "min_roughness": 0.2,
    "laplace_scale":  20000.0,
    "iter": 3000,
    "save_interval": 100,
    "texture_res": [ 2048, 2048 ],
    "train_res": [512, 512],
    "batch": 8,
    "learning_rate": [0.03, 0.03],
    "kd_min" : [0.03, 0.03, 0.03],
    "kd_max" : [0.8, 0.8, 0.8],
    "ks_min" : [0, 0.04, 0],
    "ks_max" : [0, 1.0, 1.0],
    "dmtet_grid" : 128,
    "mesh_scale" : 5,
    "camera_space_light" : true,
    "background" : "white",
    "display" : [{"bsdf":"kd"}, {"bsdf":"ks"}, {"bsdf" : "normal"}],
    "validate" : true,
    "lock_pos" : true,
    "out_dir": "3dssl/subject85.150-24-colmap-w-2mesh"}
```

### Model 26

```json
    {"ref_mesh": "data/3dssl/s850_rescaled",
    "random_textures": true,
    "sdf_regularizer": 0.01,
    "min_roughness": 0.2,
    "loss": "logl2",
    "laplace_scale":  20000.0,
    "iter": 3000,
    "save_interval": 100,
    "texture_res": [ 2048, 2048 ],
    "train_res": [512, 512],
    "batch": 8,
    "learning_rate": [0.03, 0.03],
    "kd_min" : [0.03, 0.03, 0.03],
    "kd_max" : [0.8, 0.8, 0.8],
    "ks_min" : [0, 0.04, 0],
    "ks_max" : [0, 1.0, 1.0],
    "dmtet_grid" : 128,
    "mesh_scale" : 5,
    "camera_space_light" : true,
    "background" : "white",
    "display" : [{"bsdf":"kd"}, {"bsdf":"ks"}, {"bsdf" : "normal"}],
    "validate" : true,
    "lock_pos" : true,
    "out_dir": "3dssl/subject85.150-26-colmap"}
```

### Model 30

```json
    {"ref_mesh": "data/3dssl/s850_rescaled",
    "random_textures": false,
    "sdf_regularizer": 10,
    "min_roughness": 0.2,
    "laplace_scale":  20000.0,
    "iter": 3000,
    "save_interval": 100,
    "texture_res": [ 2048, 2048 ],
    "train_res": [512, 512],
    "batch": 8,
    "learning_rate": [0.03, 0.03],
    "kd_min" : [0.03, 0.03, 0.03],
    "kd_max" : [0.8, 0.8, 0.8],
    "ks_min" : [0, 0.04, 0],
    "ks_max" : [0, 1.0, 1.0],
    "dmtet_grid" : 128,
    "mesh_scale" : 5,
    "camera_space_light" : true,
    "background" : "white",
    "display" : [{"bsdf":"kd"}, {"bsdf":"ks"}, {"bsdf" : "normal"}],
    "validate" : true,
    "lock_pos" : true,
    "out_dir": "3dssl/subject85.150-30-colmap-sdf-10"}
```

### Model 35
`RADIUS = 2.0`
```json
    {"ref_mesh": "data/3dssl/s850_rescaled",
    "random_textures": false,
    "sdf_regularizer": 50,
    "min_roughness": 0.2,
    "laplace_scale":  20000.0,
    "iter": 3000,
    "save_interval": 100,
    "texture_res": [ 2048, 2048 ],
    "train_res": [512, 512],
    "batch": 8,
    "learning_rate": [0.03, 0.03],
    "kd_min" : [0.03, 0.03, 0.03],
    "kd_max" : [0.8, 0.8, 0.8],
    "ks_min" : [0, 0.04, 0],
    "ks_max" : [0, 1.0, 1.0],
    "dmtet_grid" : 128,
    "mesh_scale" : 5,
    "camera_space_light" : true,
    "background" : "white",
    "display" : [{"bsdf":"kd"}, {"bsdf":"ks"}, {"bsdf" : "normal"}],
    "validate" : true,
    "lock_pos" : true,
    "out_dir": "3dssl/subject85.150-35-colmap-sdf-10-R2.0"}
```

### Model 38

`RADIUS = 3.0`
```json
    {"ref_mesh": "data/3dssl/s850_rescaled",
    "random_textures": false,
    "sdf_regularizer": 0.01,
    "min_roughness": 0.2,
    "laplace_scale":  20000.0,
    "iter": 1000,
    "save_interval": 100,
    "texture_res": [ 2048, 2048 ],
    "train_res": [512, 512],
    "batch": 8,
    "learning_rate": [0.03, 0.03],
    "kd_min" : [0.03, 0.03, 0.03],
    "kd_max" : [0.8, 0.8, 0.8],
    "ks_min" : [0, 0.04, 0],
    "ks_max" : [0, 1.0, 1.0],
    "dmtet_grid" : 256,
    "mesh_scale" : 1,
    "camera_space_light" : true,
    "background" : "white",
    "display" : [{"bsdf":"kd"}, {"bsdf":"ks"}, {"bsdf" : "normal"}],
    "validate" : true,
    "lock_pos" : true,
    "out_dir": "3dssl/subject85.150-48-ext-Scl-1-R3-TetMsh"}
```

### Model 83

`RADIUS = 3` and `train.py` line # 176: `mat['bsdf'] = 'diffuse'` changed it from `pbr` to `diffuse`.

```json
{
    "ref_mesh": "data/3dssl/s850_rescaled",
    "random_textures": false,
    "sdf_regularizer": 500,
    "min_roughness": 100,
    "laplace_scale":  300.0,
    "iter": 1000,
    "save_interval": 100,
    "texture_res": [ 2048, 1408 ],
    "train_res": [512, 352],
    "batch": 8,
    "learning_rate": [0.03, 0.03],
    "kd_min" : [0.03, 0.03, 0.03],
    "kd_max" : [0.8, 0.8, 0.8],
    "ks_min" : [0, 1.0, 0],
    "ks_max" : [0, 1.0, 0],
    "nrm_min": [0.0, 0.0, 0.0],
    "nrm_max": [1.0, 1.0, 1.0],
    "dmtet_grid" : 128,
    "mesh_scale" : 0.5,
    "env_scale": 2.0,
    "camera_space_light" : true,
    "background" : "white",
    "display" : [{"bsdf":"kd"}, {"bsdf":"ks"}, {"bsdf" : "normal"}],
    "validate" : true,
    "lock_pos" : true,
    "out_dir": "3dssl/subject85.150-83-ext-scl0.5-diffuse"
}

```

### Model 84

same as above with minor change in env_scale

```json
{
    "ref_mesh": "data/3dssl/s850_rescaled",
    "random_textures": false,
    "sdf_regularizer": 500,
    "min_roughness": 100,
    "laplace_scale":  300.0,
    "iter": 1000,
    "save_interval": 100,
    "texture_res": [ 2048, 1408 ],
    "train_res": [512, 352],
    "batch": 8,
    "learning_rate": [0.03, 0.03],
    "kd_min" : [0.03, 0.03, 0.03],
    "kd_max" : [0.8, 0.8, 0.8],
    "ks_min" : [0, 1.0, 0],
    "ks_max" : [0, 1.0, 0],
    "nrm_min": [0.0, 0.0, 0.0],
    "nrm_max": [1.0, 1.0, 1.0],
    "dmtet_grid" : 128,
    "mesh_scale" : 0.5,
    "env_scale": 0.5,
    "camera_space_light" : true,
    "background" : "white",
    "display" : [{"bsdf":"kd"}, {"bsdf":"ks"}, {"bsdf" : "normal"}],
    "validate" : true,
    "lock_pos" : true,
    "out_dir": "3dssl/subject85.150-84-ext-scl0.5-diffuse"
}
```

### Model 88

Increased the size of the tetra-grid to 256, down-scaled batch size to 4. Had to off-load memory to run the training.

```json
{
    "ref_mesh": "data/3dssl/s850_rescaled",
    "random_textures": false,
    "sdf_regularizer": 500,
    "min_roughness": 200,
    "laplace_scale":  300.0,
    "iter": 1000,
    "save_interval": 100,
    "texture_res": [ 2048, 1408 ],
    "train_res": [512, 352],
    "pre_load": false,
    "batch": 4,
    "learning_rate": [0.03, 0.03],
    "kd_min" : [0.03, 0.03, 0.03],
    "kd_max" : [0.8, 0.8, 0.8],
    "ks_min" : [0, 1.0, 0],
    "ks_max" : [0, 1.0, 0],
    "nrm_min": [0.0, 0.0, 0.0],
    "nrm_max": [1.0, 1.0, 1.0],
    "dmtet_grid" : 128,
    "mesh_scale" : 0.5,
    "env_scale": 0.5,
    "camera_space_light" : true,
    "background" : "white",
    "display" : [{"bsdf":"kd"}, {"bsdf":"ks"}, {"bsdf" : "normal"}],
    "validate" : true,
    "lock_pos" : true,
    "out_dir": "3dssl/subject85.150-88-ext-scl1-diffuse"
}

```

### Model 99

Downscaled the texture resolution, grid size, tet-grid size, batch size, other parameters to half what was used on 128 grid. And increased the layers to 2.

```json
{
    "ref_mesh": "data/3dssl/s850_rescaled",
    "random_textures": false,
    "layers": 2,
    "sdf_regularizer": 250,
    "min_roughness": 100,
    "laplace_scale":  60.0,
    "iter":1000,
    "save_interval": 100,
    "texture_res": [ 1024, 704 ],
    "train_res": [512, 352],
    "pre_load": false,
    "batch": 4,
    "learning_rate": [0.03, 0.03],
    "kd_min" : [0.03, 0.03, 0.03],
    "kd_max" : [0.8, 0.8, 0.8],
    "ks_min" : [0, 1.0, 0],
    "ks_max" : [0, 1.0, 0],
    "nrm_min": [0.0, 0.0, 0.0],
    "nrm_max": [1.0, 1.0, 1.0],
    "dmtet_grid" : 64,
    "mesh_scale" : 0.5,
    "env_scale": 0.5,
    "camera_space_light" : true,
    "background" : "white",
    "display" : [{"bsdf":"kd"}, {"bsdf":"ks"}, {"bsdf" : "normal"}],
    "validate" : true,
    "lock_pos" : true,
    "out_dir": "3dssl/subject85.150-99-ext-scl0.5-diffuse"
}
```

### Model 109

```json
{
    "ref_mesh": "data/3dssl/s850_rescaled",
    "random_textures": false,
    "layers": 1,
    "sdf_regularizer": 250,
    "min_roughness": 100,
    "laplace_scale":  120.0,
    "iter":3000,
    "save_interval": 100,
    "texture_res": [ 1024, 704 ],
    "train_res": [512, 352],
    "pre_load": false,
    "batch": 4,
    "learning_rate": [0.03, 0.03],
    "kd_min" : [0.03, 0.03, 0.03],
    "kd_max" : [0.8, 0.8, 0.8],
    "ks_min" : [0, 1.0, 0],
    "ks_max" : [0, 1.0, 0],
    "nrm_min": [0.0, 0.0, 0.0],
    "nrm_max": [1.0, 1.0, 1.0],
    "dmtet_grid" : 128,
    "mesh_scale" : 1.0,
    "env_scale": 1.0,
    "camera_space_light" : true,
    "background" : "white",
    "display" : [{"bsdf":"kd"}, {"bsdf":"ks"}, {"bsdf" : "normal"}],
    "validate" : true,
    "lock_pos" : true,
    "out_dir": "3dssl/subject85.150-109-ext-scl1-diffuse"
}
```

### Model 110 

Good results

```json
{
    "ref_mesh": "data/3dssl/s850_rescaled",
    "random_textures": false,
    "layers": 1,
    "sdf_regularizer": 250,
    "min_roughness": 100,
    "laplace_scale":  120.0,
    "iter":3000,
    "save_interval": 100,
    "texture_res": [ 1024, 704 ],
    "train_res": [512, 352],
    "pre_load": false,
    "batch": 4,
    "learning_rate": [0.03, 0.03],
    "kd_min" : [0.03, 0.03, 0.03],
    "kd_max" : [0.8, 0.8, 0.8],
    "ks_min" : [0, 1.0, 0],
    "ks_max" : [0, 1.0, 0],
    "nrm_min": [0.0, 0.0, 0.0],
    "nrm_max": [1.0, 1.0, 1.0],
    "dmtet_grid" : 128,
    "mesh_scale" : 0.6,
    "env_scale": 0.6,
    "camera_space_light" : true,
    "background" : "white",
    "display" : [{"bsdf":"kd"}, {"bsdf":"ks"}, {"bsdf" : "normal"}],
    "validate" : true,
    "lock_pos" : true,
    "out_dir": "3dssl/subject85.150-110-ext-scl1-diffuse"
}
```

### Model 112

```json
{
    "ref_mesh": "data/3dssl/s850_rescaled",
    "random_textures": false,
    "layers": 1,
    "sdf_regularizer": 250,
    "min_roughness": 100,
    "laplace_scale":  120.0,
    "iter":3000,
    "save_interval": 100,
    "texture_res": [ 2048, 1408 ],
    "train_res": [512, 352],
    "pre_load": false,
    "batch": 4,
    "learning_rate": [0.03, 0.03],
    "kd_min" : [0.03, 0.03, 0.03],
    "kd_max" : [0.8, 0.8, 0.8],
    "ks_min" : [0, 1.0, 0],
    "ks_max" : [0, 1.0, 0],
    "nrm_min": [0.0, 0.0, 0.0],
    "nrm_max": [1.0, 1.0, 1.0],
    "dmtet_grid" : 128,
    "mesh_scale" : 0.6,
    "env_scale": 0.6,
    "camera_space_light" : true,
    "background" : "white",
    "display" : [{"bsdf":"kd"}, {"bsdf":"ks"}, {"bsdf" : "normal"}],
    "validate" : true,
    "lock_pos" : true,
    "out_dir": "3dssl/subject85.150-112-ext-scl1-diffuse"
}
```
### Model 124

```json
{
    "ref_mesh": "data/3dssl/s850_rescaled",
    "random_textures": false,
    "spp": 1,
    "sdf_regularizer": 250,
    "min_roughness": 100,
    "laplace": "absolute",
    "laplace_scale":  120.0,
    "iter": 3000,
    "save_interval": 100,
    "texture_res": [ 2048, 1408 ],
    "train_res": [512, 352],
    "pre_load": false,
    "batch": 10,
    "learning_rate": [0.03, 0.03],
    "kd_min" : [0.03, 0.03, 0.03],
    "kd_max" : [0.8, 0.8, 0.8],
    "ks_min" : [0, 0.5, 0],
    "ks_max" : [0, 1.0, 1.0],
    "dmtet_grid" : 128,
    "mesh_scale" : 0.6,
    "env_scale": 0.6,
    "camera_space_light" : true,
    "background" : "white",
    "display" : [{"bsdf":"kd"}, {"bsdf":"ks"}, {"bsdf" : "normal"}],
    "validate" : true,
    "lock_pos" : true,
    "out_dir": "3dssl/subject85.150-124-ext.scaled-scl1-diffuse"
}
```
