# SE_SSL
# SSL-based Speech Enhancement 

This repository contains legacy (under development) code for developing a framework  SSL-based speech enhancement.

## Requirements

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Data Preparation

The dataset is expected to be in the `audio/` folder. 
- `audio/clean_testset_wav_16k` contains the clean test set.
- `audio/noisy_testset_wav_16k` contains the noisy test set.
- `audio/clean_trainset_wav_16k` contains the clean training set.
- `audio/noisy_trainset_wav_16k` contains the noisy training set.

You can update the path to the dataset in the [`configs/mpnet_weights.json file.]. To apply PCS to the audio you can use apply_pcs, please make sure to update the paths in the apply_pcs.py for audios.
The configuration file also contains other parameters that you can change to run the experiment (e.g., compression, learnable sigmoid, etc.).


## Training and evaluation

To run the experiment you can use the following command to train the model:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --experiment_config configs/classification_aligned.json \
    --num_epochs 100 \
    --batch_size 4 \
    --cuda  \
    --model_tag facebook/wav2vec2-base \
    --checkpoint_dir trained_models/wav2vec2_base/ \
    --magnitude_head conformer \
```

And to evaluate the model you can use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python compute_metrics.py \
    --experiment_config configs/masking_compression.json \
    --model_checkpoint trained_models/wav2vec2_base/best_model.pt \
    --cuda \
    --reconstructed_audio_folder reconstructed_audio/wav2vec2_base/ \
    --model_tag facebook/wav2vec2-base \
    --magnitude_head conformer  > results/wav2vec2_base.txt
```

After running the evaluation script, you can use the `results/wav2vec2_base.txt` file to check the results in terms of PESQ and STOI.

You should set the command line arguments according to your needs. For example, you can change the model tag to use a different pretrained model. You can also change the magnitude head to use different architectures (e.g., `lstm` or `transformer`). 

You should also check the `CUDA_VISIBLE_DEVICES` variable to make sure that you are using the correct GPU. The code is only tested on a single GPU at the moment.

