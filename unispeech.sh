
export CUDA_VISIBLE_DEVICES=0


mkdir -p mag_only/unispeech
mkdir -p reconstructed_audio/mag_only/unispeech
mkdir -p results/mag_only

python train_disc.py \
    --experiment_config configs/mpnet_weights.json \
    --num_epochs 50 \
    --batch_size 4\
    --cuda  \
    --model_tag microsoft/unispeech-sat-large \
    --checkpoint_dir mag_only/unispeech/ \
    --reconstructed_audio_folder reconstructed_audio/mag_only/unispeech \
    --compute_metrics_interval 1\
    --magnitude_head conformer\
    --experiment_name mag_only/metric_unispeech \
    --log_on_comet 

python compute_metrics_v3.py \
    --experiment_config configs/mpnet_weights.json \
    --model_checkpoint /home/salman/SE_Self-Supervise_Learning-/mag_only/unispeech/best_model.pt \
    --cuda \
    --reconstructed_audio_folder reconstructed_audio/mag_only/unispeech \
    --model_tag microsoft/unispeech-sat-large \
    --magnitude_head conformer  > results/mag_only/unispeech.txt





