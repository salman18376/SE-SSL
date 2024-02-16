export CUDA_VISIBLE_DEVICES=0

mkdir -p mag_only/whisper_duplication
mkdir -p reconstructed_audio/mag_only/whisper_duplication
mkdir -p results/mag_only

python train_whisper.py \
    --experiment_config configs/whisper_duplication.json \
    --num_epochs 100 \
    --batch_size 8\
    --cuda  \
    --model_tag openai/whisper-small \
    --checkpoint_dir mag_only/whisper_duplication/ \
    --reconstructed_audio_folder reconstructed_audio/mag_only/whisper_duplication \
    --compute_metrics_interval 1\
    --magnitude_head conformer\
    --phase_head conformercomplex \
    --experiment_name mag_only/whisper_duplication \
    --log_on_comet 

python compute_metrics_v3.py \
    --experiment_config configs/whisper_duplication.json \
    --model_checkpoint mag_only/whisper_duplication/best_model.pt \
    --cuda \
    --reconstructed_audio_folder reconstructed_audio/mag_only/whisper_duplication \
    --model_tag openai/whisper-large  \
    --magnitude_head conformer \
    --phase_head conformercomplex  > results/mag_only/whisper_duplication.txt


