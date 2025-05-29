import numpy as np

from mqc_pipeline import Structure


def test_Structure():
    # Test initialization with minimal parameters
    elements = ['O', 'H', 'H']
    xyz = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    st = Structure(elements=elements, xyz=xyz)
    assert st.elements == elements
    assert np.allclose(st.xyz, xyz)

    # Test that unique_id is generated and check its length
    assert st.unique_id is not None
    assert len(st.unique_id) == 16
    assert st.property == {}
    assert st.metadata == {}

    # Test the structural equality
    assert st == Structure(elements=elements, xyz=xyz)


def test_from_xyz_block():
    xyz_block_extended = """3

O      1.223982   -0.063887    0.000000 0.0
C     -0.001651    0.128033   -0.000000 0.0
O     -1.222331   -0.064146   -0.000000 0.0
"""
    # Handles extended XYZ blocks with additional columns properly
    st = Structure.from_xyz_block(xyz_block_extended)
