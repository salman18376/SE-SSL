
export CUDA_VISIBLE_DEVICES=2


mkdir -p mag_only/pcs_target_only
mkdir -p reconstructed_audio/mag_only/pcs_target_only
mkdir -p results/mag_only

python train_disc.py \
    --experiment_config configs/pcs_target_only.json \
    --num_epochs 50 \
    --batch_size 4\
    --cuda  \
    --model_tag microsoft/wavlm-large \
    --checkpoint_dir mag_only/pcs_target_only/ \
    --reconstructed_audio_folder reconstructed_audio/mag_only/pcs_target_only \
    --compute_metrics_interval 1\
    --magnitude_head conformer\
    --experiment_name mag_only/pcs_target_only \
    --log_on_comet 

python compute_metrics_v3.py \
    --experiment_config configs/pcs_target_only.json \
    --model_checkpoint /home/salman/mag_only/pcs_target_only/best_model.pt \
    --cuda \
    --reconstructed_audio_folder reconstructed_audio/mag_only/pcs_target_only \
    --model_tag microsoft/wavlm-large \
    --magnitude_head conformer \  > results/mag_only/pcs_target_only.txt





