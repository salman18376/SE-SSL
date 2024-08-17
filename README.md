# SE_SSL
# Exploiting Consistency-Preserving Loss and Perceptual Contrast Stretching to Boost SSL-based Speech Enhancement
#PCS_CS_WAVLM
## Requirements

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Data Preparation
Download the VCTK-DEMAND dataset with 16 kHz, and change the dataset dir:
The dataset is expected to be in the `audio/` folder. 
- `audio/clean_testset_wav_16k` contains the clean test set.
- `audio/noisy_testset_wav_16k` contains the noisy test set.
- `audio/clean_trainset_wav_16k` contains the clean training set.
- `audio/noisy_trainset_wav_16k` contains the noisy training set.

You can update the path to the dataset in the [`configs/SE_SSL.json file.]. 
The configuration file also contains other parameters you can change to run the experiment (e.g., compression, learnable sigmoid, etc.).

## PCS on Audios
To apply PCS to the audio you can use apply_pcs, please update the paths in the apply_pcs.py for audios.

## For best model 

Run wavlm_best.sh for the best model, which is a conformer as a head. Additionally, remember to use waveform_loss (weighted_sdr_loss) in conjunction with consistency_loss (L1) and unconsistency_loss (L1). You can also download the best model weights from https://drive.google.com/file/d/1R3XnnmFNu8xDb3oJg2Ct7BkJ9RP24Gqk/view?usp=sharing.

## Training and evaluation

To run the experiment you can use the following command to train the model:

```bash
CUDA_VISIBLE_DEVICES=0 python train_disc.py \
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
```

To evaluate the model you can use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python compute_metrics_v3.py \
    --experiment_config configs/se_ssl.json \
    --model_checkpoint /home/salman/SE_Self-Supervise_Learning-/mag_only/wavlm_best/best_model.pt \
    --cuda \
    --reconstructed_audio_folder reconstructed_audio/mag_only/wavlm_best \
    --model_tag microsoft/wavlm-large \
    --magnitude_head conformer > results/mag_only/wavlm_best.txt
```
After running the evaluation script, you can use the `results/mag_only/wavlm_best.txt` file to check the results regarding PESQ and STOI.

You can set the command line arguments according to your needs. For example, you can change the model tag to use a different pre-trained model. You can also change the magnitude head to use different architectures (e.g., `lstm` or `transformer`). 

You should also check the `CUDA_VISIBLE_DEVICES` variable to make sure that you are using the correct GPU. The code is only tested on a single GPU at the moment.

Enhanced WAV files are saved in the reconstructed_audio_folder.
For computing other metrics like CBAK, COVL, etc you can run metrics.sh and update the paths for enhanced and clean waveforms accordingly.





