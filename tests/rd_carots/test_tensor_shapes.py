"""Test tensor shapes throughout pipeline"""
import pytest
import torch
import numpy as np
from models.rd_carots.io_schema import IOSchema, split_io_variables

def test_io_split_shapes():
    io_schema = IOSchema(
        mode='explicit_io',
        input_indices=[0, 1, 2],
        output_indices=[3, 4, 5, 6],
        ignored_indices=[],
        n_inputs=3,
        n_outputs=4,
        n_total=7
    )
    
    data = np.random.randn(32, 10, 7)
    inputs, outputs = split_io_variables(data, io_schema)
    
    assert inputs.shape == (32, 10, 3)
    assert outputs.shape == (32, 10, 4)

def test_batch_metadata_shapes():
    batch_size = 16
    group_ids = torch.zeros(batch_size * 4, dtype=torch.long)
    group_ids[batch_size:2*batch_size] = 1
    group_ids[2*batch_size:3*batch_size] = 2
    group_ids[3*batch_size:] = 3
    
    assert group_ids.shape[0] == batch_size * 4
    assert (group_ids[:batch_size] == 0).all()
    assert (group_ids[batch_size:2*batch_size] == 1).all()

def test_embedding_shapes():
    batch_size = 32
    embedding_dim = 512
    embeddings = torch.randn(batch_size, embedding_dim)
    
    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    assert normalized.shape == (batch_size, embedding_dim)
    assert torch.allclose(torch.norm(normalized, p=2, dim=1), torch.ones(batch_size), atol=1e-5)
