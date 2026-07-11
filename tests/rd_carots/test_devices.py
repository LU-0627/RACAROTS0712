"""Test CPU and CUDA device handling"""
import pytest
import torch

def test_cpu_device():
    device = torch.device('cpu')
    x = torch.randn(10, 20, device=device)
    assert x.device.type == 'cpu'

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_device():
    device = torch.device('cuda:0')
    x = torch.randn(10, 20, device=device)
    assert x.device.type == 'cuda'

def test_device_transfer():
    x_cpu = torch.randn(10, 20)
    assert x_cpu.device.type == 'cpu'
    
    if torch.cuda.is_available():
        x_cuda = x_cpu.to('cuda:0')
        assert x_cuda.device.type == 'cuda'
        
        x_back = x_cuda.to('cpu')
        assert x_back.device.type == 'cpu'
        assert torch.allclose(x_cpu, x_back)
