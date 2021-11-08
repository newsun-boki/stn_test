import torch
from stn_test import Net
import torchvision
model = Net()
stat_dict = torch.load("./output/last.pt")
model.load_state_dict(stat_dict)
model.eval()
example = torch.rand((1, 1, 28, 28))
traced_script = torch.jit.trace(model, example)
traced_script.save("./output/trans/last.pt")
