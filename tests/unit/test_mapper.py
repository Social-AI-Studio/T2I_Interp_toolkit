"""Unit tests for mapper modules."""

import pytest
import torch

from t2Interp.mapper import MLPMapper, MLPMapperTwoHeads


def test_mlp_mapper_initialization():
    """Test MLPMapper can be initialized with correct dimensions."""
    input_dim = 100
    hidden_dim = 50
    output_dim = 10

    mapper = MLPMapper(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    # MLPMapper stores network as nn.Sequential, not individual dims
    assert mapper.network is not None
    assert len(list(mapper.parameters())) > 0


def test_mlp_mapper_forward_pass():
    """Test MLPMapper forward pass produces correct output shape."""
    batch_size = 4
    input_dim = 100
    hidden_dim = 50
    output_dim = 10

    mapper = MLPMapper(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    x = torch.randn(batch_size, input_dim)

    output = mapper(x)

    assert output.shape == (batch_size, output_dim)
    assert not torch.isnan(output).any(), "Output contains NaN values"


def test_mlp_mapper_gradient_flow():
    """Test that gradients flow through the mapper."""
    mapper = MLPMapper(input_dim=100, hidden_dim=50, output_dim=10)
    x = torch.randn(4, 100, requires_grad=True)

    output = mapper(x)
    loss = output.sum()
    loss.backward()

    # Check that gradients exist
    assert x.grad is not None
    for param in mapper.parameters():
        assert param.grad is not None


def test_mlp_mapper_two_heads_initialization():
    """Test MLPMapperTwoHeads initialization."""
    # Uses output_dims as a list, not output_dim_1/output_dim_2
    mapper = MLPMapperTwoHeads(
        input_dim=100,
        hidden_dim=50,
        output_dims=[7, 3],
    )

    # Check the heads exist
    assert mapper.head1 is not None
    assert mapper.head2 is not None
    assert mapper.trunk is not None


def test_mlp_mapper_two_heads_forward():
    """Test MLPMapperTwoHeads produces two outputs."""
    batch_size = 4
    mapper = MLPMapperTwoHeads(
        input_dim=100,
        hidden_dim=50,
        output_dims=[7, 3],
    )

    x = torch.randn(batch_size, 100)
    output1, output2 = mapper(x)

    assert output1.shape == (batch_size, 7)
    assert output2.shape == (batch_size, 3)


@pytest.mark.parametrize(
    "input_dim,hidden_dim,output_dim",
    [
        (128, 64, 10),
        (256, 128, 20),
        (512, 256, 5),
    ],
)
def test_mlp_mapper_various_dimensions(input_dim, hidden_dim, output_dim):
    """Test MLPMapper with various dimension configurations."""
    mapper = MLPMapper(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    x = torch.randn(2, input_dim)
    output = mapper(x)

    assert output.shape == (2, output_dim)


def test_mlp_mapper_device_placement():
    """Test that mapper can be moved to different devices."""
    mapper = MLPMapper(input_dim=100, hidden_dim=50, output_dim=10)

    # Test CPU
    mapper_cpu = mapper.to("cpu")
    x_cpu = torch.randn(2, 100)
    output_cpu = mapper_cpu(x_cpu)
    assert output_cpu.device.type == "cpu"

    # Test CUDA only if available
    if torch.cuda.is_available():
        mapper_cuda = mapper.to("cuda")
        x_cuda = torch.randn(2, 100).cuda()
        output_cuda = mapper_cuda(x_cuda)
        assert output_cuda.device.type == "cuda"


def test_mlp_mapper_device_property():
    """Test MLPMapper.device property."""
    mapper = MLPMapper(input_dim=100, hidden_dim=50, output_dim=10)
    assert mapper.device == torch.device("cpu")


def test_mlp_mapper_two_heads_device_property():
    """Test MLPMapperTwoHeads.device property."""
    mapper = MLPMapperTwoHeads(input_dim=100, hidden_dim=50, output_dims=[7, 3])
    assert mapper.device == torch.device("cpu")


def test_mlp_mapper_two_heads_with_dropout():
    """Test MLPMapperTwoHeads with dropout enabled."""
    mapper = MLPMapperTwoHeads(
        input_dim=100,
        hidden_dim=50,
        output_dims=[7, 3],
        dropout=0.1,
    )

    x = torch.randn(4, 100)
    output1, output2 = mapper(x)

    assert output1.shape == (4, 7)
    assert output2.shape == (4, 3)
