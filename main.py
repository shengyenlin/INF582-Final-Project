from modules.barthez import Barthez
import configs.barthez_config as base_config
from dataloaders.base_dataloader import base_dataloader
from train import fit

train_loader = base_dataloader(base_config.DownStream.batch_size, True, "train", base_config)
val_loader = base_dataloader(32, False, "val", base_config)

model = Barthez()
model.to(base_config.device)

model = fit(model, base_config, train_loader, val_loader)
