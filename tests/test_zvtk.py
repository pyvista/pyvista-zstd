from pathlib import Path
import pyvista as pv
import zvtk
from pyvista import examples


def test_zvtk_grid(tmp_path: Path):
    ugrid = examples.load_hexbeam()

    tmp_filename = tmp_path / "ugrid.zvtu"
    zvtk.compress(ugrid, tmp_filename)
    ugrid_out = zvtk.decompress(tmp_filename)

    assert ugrid.point_data == ugrid_out.point_data
    assert ugrid.cell_data == ugrid_out.cell_data
