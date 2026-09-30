import torch
from model import MRCNN, TCE, EncoderLayer, MultiHeadedAttention, PositionwiseFeedForward, AttnSleep
from copy import deepcopy

x = torch.randn(128, 1, 3000)   # matches the (batch, 1, 3000) tensor from data_loader/data_loaders.py

model = AttnSleep()
model.eval()

x_feat  = model.mrcnn(x)                                     # Stage 1: MRCNN
encoded = model.tce(x_feat)                                  # Stage 2: TCE
flat    = encoded.contiguous().view(encoded.shape[0], -1)    # flatten before fc
out     = model.fc(flat)                                     # Stage 3: fc

print(x.shape, x_feat.shape, encoded.shape, flat.shape, out.shape)