export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export OUTPUT_DIR="Flux-erase-dev

start=$SECONDS
python train_flux_lora.py \
    --config config/disentangle/config_explosive.yaml
end=$SECONDS
echo "Execution time: $((end - start)) seconds"

start=$SECONDS
python train_flux_lora.py \
    --config config/disentangle/config_frida.yaml
end=$SECONDS
echo "Execution time: $((end - start)) seconds"

start=$SECONDS
python train_flux_lora.py \
    --config config/disentangle/config_hug.yaml
end=$SECONDS
echo "Execution time: $((end - start)) seconds"