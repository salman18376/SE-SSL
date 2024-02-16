
export CUDA_VISIBLE_DEVICES=0


mkdir -p mag_only/pcs_noisy_only
mkdir -p reconstructed_audio/mag_only/pcs_noisy_only
mkdir -p results/mag_only

python train_disc.py \
    --experiment_config configs/pcs_noisy_only.json \
    --num_epochs 50 \
    --batch_size 4\
    --cuda  \
    --model_tag microsoft/wavlm-large \
    --checkpoint_dir mag_only/pcs_noisy_only/ \
    --reconstructed_audio_folder reconstructed_audio/mag_only/pcs_noisy_only \
    --compute_metrics_interval 1\
    --magnitude_head conformer\
    --experiment_name mag_only/pcs_noisy_only \
    --log_on_comet 

python compute_metrics_v3.py \
    --experiment_config configs/pcs_noisy_only.json \
    --model_checkpoint /home/salman/SE_Self-Supervise_Learning-/mag_only/pcs_noisy_only/best_model.pt \
    --cuda \
    --reconstructed_audio_folder reconstructed_audio/mag_only/pcs_noisy_only \
    --model_tag microsoft/wavlm-large \
    --magnitude_head conformer \  > results/mag_only/pcs_noisy_only.txt





