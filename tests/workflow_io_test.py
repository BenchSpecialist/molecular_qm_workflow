from pathlib import Path

import pytest
from mqc_pipeline.workflow.io import read_smiles, SMILES_COL_NAMES

csv_content_temp = """{COL_NAME},charge
CC[N+](CC)=S(F)F,1
FS(F)=[N+]1CCOCC1,1
C[n+]1ccn(S(=O)(=O)F)c1,1
Cc1n(S(=O)(=O)F)cc[n+]1C,1
CCC[N+]1=CC([S@@+]([O])(=O)F)=N[C@@H]1C,2
CCC[N+]1=CC([S@@+]([O])(=O)F)=N[C@H]1C,2
[O][S@@+](=O)(F)CC[N+]1=CN=C2C=CCC=C21,2

"""


@pytest.mark.parametrize('smiles_col_name', SMILES_COL_NAMES)
def test_read_smiles_csv(smiles_col_name, tmp_cwd):
    input_file = 'input.csv'
    Path(input_file).write_text(
        csv_content_temp.format(COL_NAME=smiles_col_name))
    smiles_list = read_smiles(input_file)

    assert set(smiles_list) == {
        'CC[N+](CC)=S(F)F', 'FS(F)=[N+]1CCOCC1', 'C[n+]1ccn(S(=O)(=O)F)c1',
        'Cc1n(S(=O)(=O)F)cc[n+]1C', 'CCC[N+]1=CC([S@@+]([O])(=O)F)=N[C@@H]1C',
        'CCC[N+]1=CC([S@@+]([O])(=O)F)=N[C@H]1C',
        '[O][S@@+](=O)(F)CC[N+]1=CN=C2C=CCC=C21'
    }
