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
    assert len(st.unique_id) == 11
    assert st.property == {}
    assert st.metadata == {}

    # Test the structural equality
    assert st == Structure(elements=elements, xyz=xyz)
