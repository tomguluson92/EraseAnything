export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export OUTPUT_DIR="Flux-erase-dev"

# nude
python train_flux_lora.py \
    --config config/config.yaml