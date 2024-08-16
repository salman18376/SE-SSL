
export CUDA_VISIBLE_DEVICES=3


mkdir -p mag_only/wavlm_best
mkdir -p reconstructed_audio/mag_only/wavlm_best
mkdir -p results/mag_only

python train_disc.py \
    --experiment_config configs/se_ssl.json \
    --num_epochs 50 \
    --batch_size 4\
    --cuda  \
    --model_tag microsoft/wavlm-large \
    --checkpoint_dir mag_only/wavlm_best/ \
    --reconstructed_audio_folder reconstructed_audio/mag_only/wavlm_best \
    --compute_metrics_interval 1\
    --magnitude_head conformer\
    --experiment_name mag_only/wavlm_best \
    --log_on_comet 

python compute_metrics_v3.py \
    --experiment_config configs/se_ssl.json \
    --model_checkpoint /home/salman/SE_Self-Supervise_Learning-/mag_only/wavlm_best/best_model.pt \
    --cuda \
    --reconstructed_audio_folder reconstructed_audio/mag_only/wavlm_best \
    --model_tag microsoft/wavlm-large \
    --magnitude_head conformer > results/mag_only/wavlm_best.txt




