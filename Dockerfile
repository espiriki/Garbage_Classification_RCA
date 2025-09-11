FROM pytorch/pytorch:latest

# Install everything else in one layer (fewer layers = smaller image)
RUN pip install \
    matplotlib \
    albumentations \
    torch-summary \
    wandb \
    protobuf \
    torchmetrics \
    seaborn \
    pandas \
    transformers \
    scikit-learn

# Ensure compatibility: pin numpy < 2
RUN pip install "numpy<2"