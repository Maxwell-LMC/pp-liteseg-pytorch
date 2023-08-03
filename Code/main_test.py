import pytorch_liteseg
from STDC1_pytorch_raw.model import STDC1
import torch


###initialise liteseg model
model = pytorch_liteseg.liteseg(num_classes=10, encoder=STDC1)

feature = torch.randn((1,3,512,1024))
model.eval()
with torch.inference_mode():
    output = model(feature)
    print(output.shape)

