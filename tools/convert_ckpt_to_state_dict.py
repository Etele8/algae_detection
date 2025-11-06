import os
import torch
import pathlib

class Config():
    SRC = "D:/intezet/Bogi/models/best_roi_count.pth"
    DST = "D:/intezet/Bogi/models/best_roi_count_state.pth"
src, dst = Config.SRC, Config.DST

if os.name == "nt":
    pathlib.PosixPath = pathlib.PurePosixPath

ckpt = torch.load(src, map_location="cpu", weights_only=False)

state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))

torch.save(state, dst)
print(f"Saved state_dict to: {dst}")