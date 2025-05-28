from .aquanet import Aquanet
import torch

def build_model(NUM_CLASSES, RESTORE_FROM):
    model = Aquanet(num_classes=NUM_CLASSES)

    saved_state_dict = torch.load(RESTORE_FROM)
    new_params = model.backbone.state_dict().copy()
    for key, value in saved_state_dict.items():
        if key.split(".")[0] not in ["fc"]:
            new_params[key] = value
    model.backbone.load_state_dict(new_params)

    return model


def load_model(NUM_CLASSES, RESTORE_FROM):
    model = Aquanet(num_classes=NUM_CLASSES)

    saved_state_dict = torch.load(RESTORE_FROM)
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    model.load_state_dict(saved_state_dict)

    return model