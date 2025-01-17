from absl.testing import absltest, parameterized
import os
import sys
import torch_xla
import torch_xla.core.xla_model as xm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    fully_shard,
    OffloadPolicy,
    register_fsdp_forward_method,
)

# Setup import folders.
xla_test_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(xla_test_folder)

import args_parse
import distributed_util as util

FLAGS = args_parse.parse_common_options()

class SmallNet(nn.Module):

  def __init__(self):
    super(SmallNet, self).__init__()
    self.net = nn.Linear(64, 64)

  def forward(self, x):
    return self.net(x)
  

def train_step(model, inputs, labels, optimizer, loss_fn):
  optimizer.zero_grad()
  outputs = model(inputs)
  loss = loss_fn(outputs, labels)
  loss.backward()
  optimizer.step()
  xm.mark_step()
  return loss


class TestTorchNaiveFSDP(parameterized.TestCase):

  @staticmethod
  def _fsdp_correctness(rank):
    dist.init_process_group("xla", init_method="xla://")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = xm.xla_device()

    with torch.device("meta"):
      model = SmallNet()
    # apply fsdp on model
    fully_shard(model)

    model.to_empty(device=device)
    # with torch.no_grad():
    #     model.init_weights(buffer_device=device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=1e-1)
    loss_fn = nn.MSELoss()

    local_batch_size = 2
    global_batch_size = local_batch_size * world_size
    offset = rank * local_batch_size
    rng = torch.Generator().manual_seed(2022)
    inputs = torch.randn(global_batch_size, 10, generator=rng)
    labels = torch.randn(global_batch_size, 10, generator=rng)
    loss = train_step(model, inputs, labels, optimizer, loss_fn)
    print(loss)

  def test_ddp_correctness(self):
    torch_xla.launch(self._fsdp_correctness)


if __name__ == "__main__":
  absltest.main()
