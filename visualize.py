import meshio

def save_multiple_field(path, points, faces, cell_dict):
    """
    Save the field into a vtu file

    Inputs: - saving path for the vtu file
            - vertices of the geometry
            - faces of the geometry
            - dict with fields to save
    """
    cells = {
        # "quad": faces
        "triangle": faces
    }
    meshio.write_points_cells(
        path,
        points,
        cells,
        point_data= {"triangle": cell_dict}
    )
