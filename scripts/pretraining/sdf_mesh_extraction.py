from nerfstudio.fields.sdf_field import SDFField
from nerfstudio.utils.marching_cubes import get_surface_sliding
from pathlib import Path

def extract_mesh(
        savePath: str,
        field: SDFField,
        resolution: int,
        bounding_box_min=(-1.0, -1.0, -1.0),
        bounding_box_max=(1.0, 1.0, 1.0)
):
    assert resolution % 256 == 0

    path = Path(savePath)
    path.parent.mkdir(parents=True, exist_ok=True)

    get_surface_sliding(
        sdf=lambda x: field.forward_geonetwork(x)[:, 0].contiguous(),
        bounding_box_min=bounding_box_min,
        bounding_box_max=bounding_box_max,
        output_path=path,
        simplify_mesh=False
    )