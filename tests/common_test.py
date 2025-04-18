import pytest

from mqc_pipeline import Structure


def test_Structure():
    # Test initialization with minimal parameters
    elements = ['O', 'H', 'H']
    xyz = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    st = Structure(elements=elements, xyz=xyz)
    assert st.elements == elements
    assert st.xyz == xyz

    # Test that unique_id is generated and check its length
    assert st.unique_id is not None
    assert len(st.unique_id) == 11  

