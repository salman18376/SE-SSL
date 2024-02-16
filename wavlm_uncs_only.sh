
export CUDA_VISIBLE_DEVICES=0


mkdir -p mag_only/wavlm_uncs_only
mkdir -p reconstructed_audio/mag_only/wavlm_uncs_only
mkdir -p results/mag_only

python train_disc.py \
    --experiment_config configs/mpnet_weights.json \
    --num_epochs 100 \
    --batch_size 4\
    --cuda  \
    --model_tag microsoft/wavlm-large \
    --checkpoint_dir mag_only/wavlm_uncs_only/ \
    --reconstructed_audio_folder reconstructed_audio/mag_only/wavlm_uncs_only \
    --compute_metrics_interval 1\
    --magnitude_head conformer\
    --experiment_name mag_only/metric_wavlm_uncs_only \
    --log_on_comet 

python compute_metrics_v3.py \
    --experiment_config configs/mpnet_weights.json \
    --model_checkpoint mag_only/wavlm_uncs_only/epoch_50.pt \
    --cuda \
    --reconstructed_audio_folder reconstructed_audio/mag_only/wavlm_uncs_only \
    --model_tag microsoft/wavlm-large \
    --magnitude_head conformer   > results/mag_only/wavlm_uncs_only.txt





