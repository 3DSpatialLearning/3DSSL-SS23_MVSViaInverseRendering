import json

def change_translation_sdfstudio(x_t: float, y_t: float, z_t: float, scale: float,
                                 path: str, fName: str) -> None:
    """
        Changes camera translation in sdfstudio meta_data.json file.
    """

    file_old = '/'.join([path, fName])
    file_new = '/'.join([path, 'n_' + fName])

    with open(file_old, 'r') as f:
        content = f.read()
        j_cont = json.loads(content)
        
        # Change translation for each frame
        for element in j_cont['frames']:
            cam2world = element['camtoworld']

            cam2world[0][3] *= scale
            cam2world[1][3] *= scale
            cam2world[2][3] *= scale

            cam2world[0][3] += x_t
            cam2world[1][3] += y_t
            cam2world[2][3] += z_t

        with open(file_new, 'w') as f_n:
            json.dump(j_cont, f_n, sort_keys=True, indent=4)


def change_translation_nerfstudio(x_t: float, y_t: float, z_t: float, scale: float, path: str, fName: str) -> None:
    """
        Change camera translation in nerfstudio transforms.json
    """
    file_old = '/'.join([path, fName])
    file_new = '/'.join([path, 'n_' + fName])

    with open(file_old, 'r') as f:
        content = f.read()
        j_cont = json.loads(content)
        
        # Change translation for each frame
        for element in j_cont['frames']:
            cam2world = element['transform_matrix']

            cam2world[0][3] *= scale
            cam2world[1][3] *= scale
            cam2world[2][3] *= scale

            cam2world[0][3] += x_t
            cam2world[1][3] += y_t
            cam2world[2][3] += z_t

        with open(file_new, 'w') as f_n:
            json.dump(j_cont, f_n, sort_keys=True, indent=4)


if __name__ == "__main__":
    change_translation_sdfstudio(0.5, 0.5, 1.0, 1.0, '../data/head_01', 'meta_data.json')