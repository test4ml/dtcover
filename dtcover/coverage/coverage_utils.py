import torch

def scale(intermediate_layer_output: torch.Tensor, rmax: float=1.0, rmin: float=0.0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled

def scale_batch(intermediate_layer_output: torch.Tensor, rmax: float=1.0, rmin: float=0.0):
    original_size = intermediate_layer_output.size()
    layer_view = intermediate_layer_output.view(intermediate_layer_output.size(0), -1)

    min_layer = layer_view.min(dim=1, keepdim=True)[0]
    max_layer = layer_view.max(dim=1, keepdim=True)[0]
    X_std = (layer_view - min_layer) / (max_layer - min_layer)
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled.view(*list(original_size))

def scale_batch_dim(intermediate_layer_output: torch.Tensor, rmax: float=1.0, rmin: float=0.0):
    original_size = intermediate_layer_output.size()
    layer_view = intermediate_layer_output.view(intermediate_layer_output.size(0), intermediate_layer_output.size(1), -1)

    min_layer = layer_view.min(dim=2, keepdim=True)[0]
    max_layer = layer_view.max(dim=2, keepdim=True)[0]
    X_std = (layer_view - min_layer) / (max_layer - min_layer)
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled.view(*list(original_size))

def scale_batch_clean(intermediate_layer_output: torch.Tensor, rmax: float=1.0, rmin: float=0.0, cleanv: float=2):
    intermediate_layer_output_clean = intermediate_layer_output.clone()
    intermediate_layer_output_clean[intermediate_layer_output_clean > cleanv] = 0
    intermediate_layer_output_clean[intermediate_layer_output_clean < -cleanv] = 0

    original_size = intermediate_layer_output.size()
    layer_view = intermediate_layer_output.view(intermediate_layer_output.size(0), -1)
    layer_view_clean = intermediate_layer_output_clean.view(intermediate_layer_output.size(0), -1)

    min_layer = layer_view_clean.min(dim=1, keepdim=True)[0]
    max_layer = layer_view_clean.max(dim=1, keepdim=True)[0]
    X_std = (layer_view - min_layer) / (max_layer - min_layer)
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled.view(*list(original_size))