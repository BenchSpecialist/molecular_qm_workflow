import json
import pytest
from unittest.mock import patch
from typing import List, Dict, Any

from ase import Atoms
from ase.io.jsonio import MyEncoder

from mqc_pipeline.batch_optimize import request_async, optimize_sts_async
from mqc_pipeline.common import Structure


class MockResponse:
    """Mock response object for aiohttp ClientSession"""

    def __init__(self, status: int, json_data: Dict[str, Any]):
        self.status = status
        self._json_data = json_data

    async def json(self) -> Dict[str, Any]:
        return self._json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def raise_for_status(self):
        if self.status != 200:
            raise Exception(f"HTTP Error {self.status}")


@pytest.fixture
def test_atoms() -> List[Atoms]:
    """Create mock ASE Atoms objects for testing"""
    return [
        Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
        Atoms('CO2', positions=[[0, 0, 0], [0, 0, 1], [0, 0, -1]]),
        Atoms('CH4',
              positions=[[0.0, 0.0, 0.0], [0.629, 0.629, 0.629],
                         [-0.629, -0.629, 0.629], [0.629, -0.629, -0.629],
                         [-0.629, 0.629, -0.629]]),
        Atoms('N2', positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]])
    ]


@pytest.fixture
def test_structures(test_atoms) -> List[Structure]:
    """Create mock Structure objects for testing"""
    return [Structure.from_ase_atoms(atom) for atom in test_atoms]


@pytest.mark.asyncio
async def test_request_async(test_atoms):
    """Test successful API call with request_async"""
    # Encode the atoms for the mock response
    atoms_data = json.dumps(test_atoms, cls=MyEncoder)

    # Create mock response
    mock_response = MockResponse(status=200, json_data={"atoms": atoms_data})

    # Patch the aiohttp ClientSession
    with patch('aiohttp.ClientSession.post', return_value=mock_response):
        result = await request_async(test_atoms, url='http://test-url')

        # Check that we got the right result
        assert result is not None
        assert len(result) == len(test_atoms)
        assert isinstance(result[0], Atoms)


async def _request_async_side_effect(batch,
                                     url,
                                     timeout=300,
                                     max_retries=3,
                                     retry_delay=5,
                                     params={}):
    return batch  # Return the batch instead of a future with single item


@pytest.mark.parametrize(
    "batch_size, call_count",
    [
        (None, 1),  # All structures are sent in one request
        (1, 4),  # Each structure is sent in a separate request
        (3, 2),  # Two requests (3 structures in first, 1 in second)
    ])
@pytest.mark.asyncio
async def test_optimize_sts_async_with_batch(test_structures, batch_size,
                                             call_count):
    with patch('mqc_pipeline.batch_optimize.request_async',
               autospec=True) as mock_request:
        mock_request.side_effect = _request_async_side_effect

        result = await optimize_sts_async(test_structures,
                                          batch_size=batch_size)

        # Verify request_async was called once for each batch
        assert mock_request.call_count == call_count

        # Check results
        assert len(result) == len(test_structures)
        assert all(isinstance(st, Structure) for st in result)
