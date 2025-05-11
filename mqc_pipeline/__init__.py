import logging

# Configure logging once when the package is imported
logging.basicConfig(filename='mqc_pipeline.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')  # excludes milliseconds

from .common import Structure
from .smiles_util import smiles_to_3d_structures_by_rdkit
