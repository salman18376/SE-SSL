
export CUDA_VISIBLE_DEVICES=0


mkdir -p mag_only/tera_logMelBase_T_F_M_AdamW_b32_1m_960hr_drop1
mkdir -p reconstructed_audio/mag_only/tera_logMelBase_T_F_M_AdamW_b32_1m_960hr_drop1
mkdir -p results/mag_only

python train_s3prl.py \
    --experiment_config configs/tera.json \
    --num_epochs 150 \
    --batch_size 32\
    --cuda  \
    --model_tag microsoft/wavlm-large \
    --checkpoint_dir mag_only/tera_logMelBase_T_F_M_AdamW_b32_1m_960hr_drop1/ \
    --reconstructed_audio_folder reconstructed_audio/mag_only/tera_logMelBase_T_F_M_AdamW_b32_1m_960hr_drop1 \
    --compute_metrics_interval 1\
    --magnitude_head conformer_tf\
    --experiment_name mag_only/metric_tera_logMelBase_T_F_M_AdamW_b32_1m_960hr_drop1 \
    --log_on_comet 

python compute_metrics_v3.py \
    --experiment_config configs/tera.json \
    --model_checkpoint /home/salman/SE_Self-Supervise_Learning-/mag_only/tera_logMelBase_T_F_M_AdamW_b32_1m_960hr_drop1/best_model.pt \
    --cuda \
    --reconstructed_audio_folder reconstructed_audio/mag_only/tera_logMelBase_T_F_M_AdamW_b32_1m_960hr_drop1 \
    --model_tag microsoft/wavlm-large \
    --magnitude_head conformer \ > results/mag_only/tera_logMelBase_T_F_M_AdamW_b32_1m_960hr_drop1.txt



