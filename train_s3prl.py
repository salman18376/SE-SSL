import comet_ml
from comet_ml import Experiment

import os
import time
import json

from tqdm import tqdm
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchaudio
from transformers import AutoFeatureExtractor, AutoModel

from data.compressor import SpectrogramCompressor
from data.audio_dataset import AudioDataset
from model.losses import weighted_sdr_loss
from data.audio_processor import AudioProcessor
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from utils import get_pred_waveforms, get_target_waveforms
from model.speech_enhamcement_s3prl import SpeechEnhancementModel
from model.discriminator import batch_pesq, MetricDiscriminator
from model.metric_loss import PerceptualLoss
import argparse
import torchmetrics
from model.s3prl import get_s3prl
from torch.nn.parallel import DataParallel
from s3prl.s3prl.nn import S3PRLUpstream

from s3prl.s3prl import hub

parser = argparse.ArgumentParser()

# ------------------------------------------------ Experiment configuration ------------------------------------------------

parser.add_argument("--experiment_config", type=str, default="configs/w2v2.json", help="path to experiment configuration file")


# ------------------------------------------------ Training parameters ------------------------------------------------

parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs to train for")
parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
parser.add_argument("--patience_early_stopping", type=int, default=20, help="number of epochs after which to stop training if validation loss does not decrease")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for training")
parser.add_argument("--num_workers", type=int, default=24, help="number of workers for data loading")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="ratio of total steps to warmup for")
parser.add_argument("--reconstructed_audio_folder", type=str, help="path to the folder where the reconstructed audio files will be saved", required=True)
parser.add_argument("--compute_metrics_interval", type=int, default=1, help="Frequency of computing metrics (in epochs)")
parser.add_argument("--cuda", action="store_true", help="use cuda")
parser.add_argument("--log_on_comet", action="store_true", help="log training on comet")
parser.add_argument("--experiment_name", type=str, default=None, help="name of the experiment on comet.ml")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory to save the model checkpoints")
parser.add_argument("--gradient_clipping", type=float, default=1.0, help="gradient clipping value")
parser.add_argument("--gradient_accumulation", type=float, default=1.0, help="gradient clipping value")
parser.add_argument("--max_model_checkpoints", type=int, default=3, help="maximum number of model checkpoints to keep")
parser.add_argument("--project_name", type=str, default="SSL_SE", help="name of the project on comet.ml")
parser.add_argument("--adaptive_lr", action="store_true", help="use adaptive learning rate, when the validation loss does not decrease, halve the learning rate")

# ------------------------------------------------ Model parameters ------------------------------------------------

parser.add_argument("--model_tag", type=str, help="tag of the transformer model to use from the HuggingFace model hub")
parser.add_argument("--freeze_ssl", action="store_true", help="freeze the weights of the SSL model")
parser.add_argument("--magnitude_head", type=str, default=None, help="head to use for the magnitude spectrogram")
parser.add_argument("--phase_head", type=str, default=None, help="head to use for the phase spectrogram")
parser.add_argument("--complex_head", type=str, default=None, help="head to use for the complex spectrogram")

parser.add_argument("--debug", action="store_true", help="debug mode")


args = parser.parse_args()


# ------------------------------------------------
# 
#              Setting the parameters 
#
# ------------------------------------------------

with open(args.experiment_config) as f:
    config = json.load(f)


if torch.cuda.is_available() and args.cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ------------------------------------------------
# 
#              Setting the data
#
# ------------------------------------------------
# Create the dataset for the noisy audio files
test_dataset = AudioDataset(
    noisy_path=config["noisy_test_folder"],
    clean_path=config["clean_test_folder"],
    n_fft=config["n_fft"],
    hop_length=config["hop_length"],
    win_length=config["win_length"],
    ssl_model_tag=args.model_tag,
    max_length_in_seconds = config["max_length_in_seconds"],
    randomly_sample_n_samples= config["randomly_sample_n_samples_testset"],
    normalize_waveform = config["normalize_waveform"],
)
train_dataset = AudioDataset(
    noisy_path=config["noisy_train_folder"],
    clean_path=config["clean_train_folder"],
    n_fft=config["n_fft"],
    hop_length=config["hop_length"],
    win_length=config["win_length"],
    ssl_model_tag=args.model_tag,
    max_length_in_seconds=config["max_length_in_seconds"],
    randomly_sample_n_samples= config["randomly_sample_n_samples"],
    normalize_waveform = config["normalize_waveform"],
)
print(f"Training dataset size: {len(train_dataset)}")



train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


# ------------------------------------------------
# 
#              Setting the model
#
# ------------------------------------------------


ssl_model = hub.tera_logMelBase_T_F_M_AdamW_b32_1m_960hr_drop1().cuda() 
print(f'Number of trainable parameters: {sum(p.numel() for p in ssl_model.parameters() if p.requires_grad) / 1e6:.2f}M')                                      


if args.log_on_comet:
    experiment = Experiment(
        api_key=os.environ["COMET_API_KEY"], 
        workspace=os.environ["COMET_WORKSPACE"],
        project_name=args.project_name,
    )
    experiment.log_parameters(config)
    experiment.log_parameters(vars(args))
    experiment_name = args.experiment_name if args.experiment_name is not None else args.experiment_config.split("/")[-1].split(".")[0]
    experiment.set_name(experiment_name)
else:
    experiment = None

if config["use_compression"]:
    compressor = SpectrogramCompressor(
        alpha=config["compressor_alpha"], 
        learnable=config["learnable_compression"], 
        compressor_type=config["compression_type"]
    )
else: compressor = None

# Initialize the speech enhancement model
model = SpeechEnhancementModel(
    ssl_model=ssl_model,
    freeze_ssl=args.freeze_ssl,
    magnitude_head=args.magnitude_head,
    phase_head=args.phase_head,
    complex_head=args.complex_head,
    ssl_embedding_dim=768,#ssl_model.config.hidden_size,NOTE: its hardcoded for s3prl
    stft_embedding_dim=config["stft_embedding_dim"],
    type=config["type"],
    sigmoid_type=config["sigmoid_type"],
    compressor=compressor,
    use_all_layers=config["use_all_layers"],
).to(device)

# If  multiple GPUs, you can use DataParallel
if torch.cuda.device_count() > 1:
    model = DataParallel(model)

audio_processor = AudioProcessor(
    sample_rate=config["sample_rate"],
    max_length_in_seconds=config["max_length_in_seconds"],
    n_fft=config["n_fft"],
    hop_length=config["hop_length"],
    win_length=config["win_length"],
    normalized_stft = config["normalized_stft"],)

print("SpeechEnhancementModel model loaded.")


# ------------------------------------------------
# 
#             Setting the training
#
# ------------------------------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


metric_discriminator = MetricDiscriminator().to(device)
metric_discriminator_optimizer = optim.Adam(metric_discriminator.parameters(), lr=args.learning_rate)

if config["scheduler"] is None:
    scheduler = None
elif config["scheduler"] == "linear_with_warmup":
    total_steps = len(train_dataloader) * args.num_epochs
    lr_lambda = lambda step: max(
        1e-9, 
        min(
            1.0, 
            step / (total_steps * args.warmup_ratio)) * (1.0 - (step - total_steps * args.warmup_ratio) / (total_steps * (1 - args.warmup_ratio))
            ) ** 0.9
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)




# ------------------------------------------------
#
#             Training the model
#
# ------------------------------------------------

# Initialize the best train loss to infinity
best_train_loss = float("inf")
early_stopping_counter = 0
if os.path.exists(args.checkpoint_dir) is False:
    os.makedirs(args.checkpoint_dir)

perceptual_loss = PerceptualLoss(model_type='wavlm', pretrained_model_name=args.model_tag)



losses = {
    "complex_loss": [],
    "phase": [],
    "waveform": [(weighted_sdr_loss, config["waveform_loss_weight"])],
    "consistency": [(nn.L1Loss(), config["consistency_loss_weight"])],
    "ssl_base_loss":[],
    "mrstft" : [],
    "magnitude" : [(nn.L1Loss(), config["consistency_loss_weight"])]
    
}

def train_step(model, 
               train_dataloader, 
               losses, optimizer, 
               audio_processor,
               compressor,
               metric_discriminator,
               metric_discriminator_optimizer,
               scheduler, 
               device, 
               magnitude_head, 
               phase_head,complex_head, 
               epoch_num, experiment=None,
               ):
    """
    Train the ssl model using the provided data and loss functions.

    Args:
        model: The ssl model to be trained.
        train_dataloader: The data loader for training data.
        losses: Dictionary of loss functions for different components.
        optimizer: The optimizer used for model parameter updates.
        scheduler: Learning rate scheduler (optional).
        device: The device to run the training on (e.g., "cuda" for GPU, "cpu" for CPU).
        compressor: Data compression utility (if used).
        audio_processor: Audio processing functions (e.g., spectrogram-to-waveform).
        epoch_num: The current epoch number for tracking progress.
        magnitude_head : magnitude head e.g conformer
        phase_head : phase head e.g conformer
        complex_head : complex head 
        metric_discriminator : metric_discriminator GAN
        metric_discriminator_optimizer : metric_discriminator_optimizer e.g Adam

    Returns:
        The average training loss for the epoch.
    """    
    
    metric_discriminator.train()
    model.train()
    train_loss = 0.0
    
    # Create a progress bar to track training progress
    p_bar = tqdm(train_dataloader, desc=f"Training (epoch {epoch_num+1})")
    for batch in p_bar:
        # Move the batch to the specified device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Zero out gradients for  the  model 
        optimizer.zero_grad()
        

        # Apply compression to the input magnitude spectrogram
        input_mag = batch["input_stft"]
        
        input_complex = batch["input_complex"]
        input_complex = input_complex.permute(0, 2, 1, 3)
        
        if compressor is not None:
            input_mag = compressor.compress(input_mag)
            batch["input_stft"] = input_mag

        predicted_mag, predicted_phase= model(batch)
        
        if config["type"] == "masking" and magnitude_head is not None:
            
            predicted_mag = predicted_mag * input_mag

        # # ------------------- ssl_base_loss -------------------
        input_waveforms = batch["input_waveform"]
        targ_waveforms = batch["output_waveform"]
        input_phase = batch["input_phase"]
        target_mag = batch["output_stft"]
        target_phase = batch["output_phase"]
        
        predicted_waveforms =get_pred_waveforms(predicted_magnitude =predicted_mag,
                                                predicted_phase =predicted_phase, 
                                                compressor=compressor,
                                                audio_processor =audio_processor, 
                                                input_waveforms =input_waveforms, 
                                                norm_factors=None)
                   

        # Compress the target spectrogram if compression is enabled
        if compressor is not None:
            target_mag = compressor.compress(target_mag) 
             
        
        target_waveforms =get_target_waveforms(targ_waveforms, 
                                                    target_mag, 
                                                    target_phase, 
                                                    audio_processor,
                                                    compressor=compressor)
        
        
         
        ssl_losses = []
        for loss_fn, loss_w in losses["ssl_base_loss"]:
           
            l = loss_fn(predicted_waveforms, target_waveforms) * loss_w
            ssl_losses.append(l)
            

        ssl_loss = sum(ssl_losses) / len(ssl_losses) if len(ssl_losses) > 0 else torch.tensor(0.0).to(device)


       # -------------------  cosine_loss -------------------
        input_waveforms = batch["input_waveform"]
        target_waveforms = batch["output_waveform"]
        target_mag = batch["output_stft"]
        target_phase = batch["output_phase"]
        output_complex = batch["output_complex"]
        output_complex = output_complex.permute(0, 2, 1, 3)
      
        cosin_losses = []
        for loss_fn, loss_w in losses["complex_loss"]:
            l = loss_fn(predicted_phase, target_mag, target_phase) * loss_w
            cosin_losses.append(l)

        cosin_loss = sum(cosin_losses) / len(cosin_losses) if len(cosin_losses) > 0 else torch.tensor(0.0).to(device)


        # -------------------  waveform loss -------------------
        
        input_waveforms = batch["input_waveform"]
        targ_waveforms = batch["output_waveform"]
        input_phase = batch["input_phase"]
        target_mag = batch["output_stft"]
        target_phase = batch["output_phase"]
        
         
 
        
        predicted_waveforms =get_pred_waveforms(predicted_magnitude =predicted_mag,
                                                predicted_phase =predicted_phase, 
                                                compressor=compressor,
                                                audio_processor =audio_processor, 
                                                input_waveforms =input_waveforms, 
                                                norm_factors=None)
                   

        # Compress the target spectrogram if compression is enabled
        if compressor is not None:
            target_mag = compressor.compress(target_mag) 
             
        
        target_waveforms =get_target_waveforms(targ_waveforms, 
                                                    target_mag, 
                                                    target_phase, 
                                                    audio_processor,
                                                    compressor=compressor)
            
        

        waveform_losses = []
        for loss_fn, loss_w in losses["waveform"]:
            try:
                l = loss_fn(input_waveforms, predicted_waveforms, target_waveforms) * loss_w
            except:
                l = loss_fn(predicted_waveforms, target_waveforms) * loss_w
            waveform_losses.append(l)

        waveform_loss = sum(waveform_losses) / len(waveform_losses) if len(waveform_losses) > 0 else torch.tensor(0.0).to(device)
        
         # -------------------  mrstft loss -------------------
        
        input_waveforms = batch["input_waveform"]
        targ_waveforms = batch["output_waveform"]
        target_mag = batch["output_stft"]
        target_phase = batch["output_phase"]
        
        
        predicted_waveforms =get_pred_waveforms(predicted_magnitude =predicted_mag,
                                                predicted_phase =predicted_phase, 
                                                compressor=compressor,
                                                audio_processor =audio_processor, 
                                                input_waveforms =input_waveforms, 
                                                norm_factors=None)
                   
           
        # Compress the target spectrogram if compression is enabled
        if compressor is not None:
            target_mag = compressor.compress(target_mag)  

        target_waveforms =get_target_waveforms(targ_waveforms, 
                                                    target_mag, 
                                                    target_phase, 
                                                    audio_processor,
                                                    compressor=compressor)
        
        
        mrstft_losses = [] 
        
        for loss_fn, loss_w in losses["mrstft"]:
 
            
            l = loss_fn(predicted_waveforms, target_waveforms) * loss_w
            mrstft_losses.append(l)

        mrstft_loss = sum(mrstft_losses) / len(mrstft_losses) if len(mrstft_losses) > 0 else torch.tensor(0.0).to(device)
        
        # -------------------  consistency_loss -------------------
        input_waveforms = batch["input_waveform"]
        targ_waveforms = batch["output_waveform"]
        target_mag = batch["output_stft"]
        target_phase = batch["output_phase"]
        
        
        predicted_waveforms =get_pred_waveforms(predicted_magnitude =predicted_mag,
                                                predicted_phase =predicted_phase, 
                                                compressor=compressor,
                                                audio_processor =audio_processor, 
                                                input_waveforms =input_waveforms, 
                                                norm_factors=None)
        
        predicted_mag = audio_processor.get_spectrogram_from_waveform(predicted_waveforms)

        # If compression is enabled, compress the predicted spectrogram
        if compressor is not None:
            predicted_mag = predicted_mag[0]  # Extract the tensor from the tuple
            predicted_mag = compressor.compress(predicted_mag)
            

            
        # Compress the target spectrogram if compression is enabled
        if compressor is not None:
            target_mag = compressor.compress(target_mag)  
            
        target_waveforms =get_target_waveforms(targ_waveforms, 
                                                    target_mag, 
                                                    target_phase, 
                                                    audio_processor,
                                                    compressor=compressor)
        target_mag = audio_processor.get_spectrogram_from_waveform(target_waveforms)

        # Compress the predicted spectrogram if compression is enabled
        if compressor is not None:
            target_mag = target_mag[0]  # Extract the tensor from the tuple
            target_mag = compressor.compress(target_mag)
            
        consistency_losses = []
        for loss_fn, loss_w in losses["consistency"]:
           
            l = loss_fn(predicted_mag, target_mag) * loss_w
            consistency_losses.append(l)

        consistency_loss_mag = sum(consistency_losses) / len(consistency_losses) if len(consistency_losses) > 0 else torch.tensor(0.0).to(device)
        
        
        
        
        # -------------------  magnitude_loss -------------------
        input_waveforms = batch["input_waveform"]
        target_mag = batch["output_stft"]

            
    
        # Compress the target spectrogram if compression is enabled
        if compressor is not None:
            target_mag = compressor.compress(target_mag)  
            
       
            
        magnitude_losses = []
        for loss_fn, loss_w in losses["magnitude"]:
           
            l = loss_fn(predicted_mag, target_mag) * loss_w
            magnitude_losses.append(l)

        unconsistency_loss_mag = sum(magnitude_losses) / len(magnitude_losses) if len(magnitude_losses) > 0 else torch.tensor(0.0).to(device)

        # -------------------  phase loss -------------------
        input_waveforms = batch["input_waveform"]
        target_waveforms = batch["output_waveform"]
        target_mag = batch["output_stft"]
        target_phase = batch["output_phase"]
        target_complex = batch["output_complex"]
         
        target_complex = target_complex.permute(0, 2, 1, 3)

        phase_losses = []
        for loss_fn, loss_w in losses["phase"]:
            
            l = loss_fn ( predicted_phase, target_phase) * loss_w
            
            phase_losses.append(l)
        
        phase_loss = sum(phase_losses) / len(phase_losses) if len(phase_losses) > 0 else torch.tensor(0.0).to(device)
        
        
        # ------------------- MetricGAN Loss -------------------
        
        if config["metric_loss_weight"] > 0.0:
            input_waveforms = batch["input_waveform"]
            target_mag = batch["output_stft"]
            targ_waveforms = batch["output_waveform"]
            target_phase = batch["output_phase"]

            

            predicted_waveforms =get_pred_waveforms(predicted_magnitude =predicted_mag,
                                                predicted_phase =predicted_phase, 
                                                compressor=compressor,
                                                audio_processor =audio_processor, 
                                                input_waveforms =input_waveforms, 
                                                norm_factors=None)
            
            

            # Compress the target spectrogram if compression is enabled
            if compressor is not None:
                target_mag = compressor.compress(target_mag)

            

            # Calculate PESQ score between target and predicted waveforms
            audio_list_r, audio_list_g = list(target_waveforms.detach().cpu().numpy()), list(predicted_waveforms.detach().cpu().numpy())
            batch_pesq_score = batch_pesq(audio_list_r, audio_list_g)

                     
            # Get the batch size for later use (creates a tensor of ones with the same batch size)
            batch_size = target_mag.shape[0]
            one_labels = torch.ones(batch_size).to(device)
            
             # Discriminator
            metric_discriminator_optimizer.zero_grad()
            
            # Calculate MetricGAN losses
            # Calculate the discriminator response for both real and generated target magnitude
            metric_r = metric_discriminator(target_mag, target_mag)
            metric_g = metric_discriminator(target_mag, predicted_mag.detach())
            loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
            
            if batch_pesq_score is not None:
                loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
            else:
                loss_disc_g = 0
            
            loss_disc_all = loss_disc_r + loss_disc_g
            
        

            # Apply gradient clipping to the model's parameters if specified
            if args.gradient_clipping is not None:
                nn.utils.clip_grad_norm_(metric_discriminator.parameters(), args.gradient_clipping)
             
            loss_disc_all.backward()
            metric_discriminator_optimizer.step()
                
            # Metric Loss TODO: try with compress predicted and targ mag
            metric_g = metric_discriminator(target_mag, predicted_mag)
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels.float())* config["metric_loss_weight"]
        else:
            loss_metric =  torch.tensor(0.0).to(device)
        
              

        # Calculate the final loss as a combination of various component losses
        final_loss = cosin_loss + consistency_loss_mag + phase_loss + waveform_loss + mrstft_loss + loss_metric + unconsistency_loss_mag  + torch.abs(ssl_loss).mean() #TODO: remove it
        # Apply gradient clipping to the model's parameters if specified
        if args.gradient_clipping is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)

        # Backpropagate the final loss through the model to update its parameters
        final_loss.backward()
        # Step the main model's optimizer to update its parameters
        optimizer.step()
             
        train_loss += final_loss.item()
        p_bar.set_postfix_str(f"Training loss: {final_loss.item():4f}")

        if experiment is not None:
            experiment.log_metric("instant_train_loss", final_loss.item(), step=epoch * len(train_dataloader) + p_bar.n)
            if scheduler is not None:
                experiment.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch * len(train_dataloader) + p_bar.n)
            else:
                experiment.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=epoch * len(train_dataloader) + p_bar.n)

            experiment.log_metric("cosin_loss", cosin_loss.item(), step=epoch * len(train_dataloader) + p_bar.n)
            experiment.log_metric("consistency_loss_mag", consistency_loss_mag.item(), step=epoch * len(train_dataloader) + p_bar.n)
            experiment.log_metric("unconsistency_loss_mag", unconsistency_loss_mag.item(), step=epoch * len(train_dataloader) + p_bar.n)
            experiment.log_metric("phase_loss", phase_loss.item(), step=epoch * len(train_dataloader) + p_bar.n)
            experiment.log_metric("waveform_loss", waveform_loss.item(), step=epoch * len(train_dataloader) + p_bar.n)
            experiment.log_metric("mrstft_loss", mrstft_loss.item(), step=epoch * len(train_dataloader) + p_bar.n)
            experiment.log_metric("ssl_loss", ssl_loss.item(), step=epoch * len(train_dataloader) + p_bar.n)
            experiment.log_metric("discriminator_loss", loss_metric.item(), step=epoch * len(train_dataloader) + p_bar.n) 

    return train_loss / len(train_dataloader)


best_model = None
model_checkpoints = {}

def save_model_if_needed(model, compressor, model_checkpoints, best_train_loss, epoch, args):
    if train_loss < max(model_checkpoints.keys(), default=1e9):
        # save the model checkpoint
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}.pt"))
        # Save compressor
        if compressor is not None:
            print(f"Saving compressor")
            torch.save(compressor.state_dict(), os.path.join(args.checkpoint_dir, f"compressor_epoch_{epoch+1}.pt"))

        # add the model checkpoint to the dictionary
        model_checkpoints[train_loss] = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}.pt")
        # remove the worst model checkpoint if the maximum number of checkpoints is reached
        if len(model_checkpoints) > args.max_model_checkpoints:
            worst_model_checkpoint = max(model_checkpoints.keys())
            os.remove(model_checkpoints[worst_model_checkpoint])
            # remove compressor checkpoint if needed
            if compressor is not None:
                compressor_filename = model_checkpoints[worst_model_checkpoint].replace("epoch", "compressor_epoch")
                os.remove(compressor_filename)
            del model_checkpoints[worst_model_checkpoint]
            

def compute_metrics(clean_dir, enhanced_dir):
    pesq_metric = torchmetrics.audio.pesq.PerceptualEvaluationSpeechQuality(config["sample_rate"], 'wb')
    stoi_metric = torchmetrics.audio.stoi.ShortTimeObjectiveIntelligibility(config["sample_rate"], False)
    
    pesq_scores = []
    stoi_scores = []
        
    def get_scores(clean_dir, enhanced_dir, filename):
        clean_path = os.path.join(clean_dir, filename)
        enhanced_path = os.path.join(enhanced_dir, filename)
        clean_waveform, _ = torchaudio.load(clean_path)
        enhanced_waveform, _ = torchaudio.load(enhanced_path)

        # check if the lengths are the same - in case trim
        if clean_waveform.shape[1] != enhanced_waveform.shape[1]:
            print(f"Lengths of clean and enhanced waveforms are different for {filename}. Trimming...")
            print(f"Clean waveform length: {clean_waveform.shape[1]}")
            print(f"Enhanced waveform length: {enhanced_waveform.shape[1]}")
            min_length = min(clean_waveform.shape[1], enhanced_waveform.shape[1])
            clean_waveform = clean_waveform[:, :min_length]
            enhanced_waveform = enhanced_waveform[:, :min_length]
        try:
            pesq_score = pesq_metric(enhanced_waveform, clean_waveform)
            stoi_score = stoi_metric(enhanced_waveform, clean_waveform)
        except Exception as e: 
            print(e)
            print(filename)
            pesq_score = 0
            stoi_score = 0
        
        return pesq_score, stoi_score
    
    
    # parallelize the computation of the metrics
    from joblib import Parallel, delayed
    # tqdm is used to show the progress bar
    scores = Parallel(n_jobs=16)(delayed(get_scores)(clean_dir, enhanced_dir, filename) for filename in tqdm(os.listdir(clean_dir), desc="Computing metrics"))
    pesq_scores = [score[0] for score in scores]
    stoi_scores = [score[1] for score in scores]

    avg_pesq = sum(pesq_scores) / len(pesq_scores)
    avg_stoi = sum(stoi_scores) / len(stoi_scores)
    
    
    
    return avg_pesq, avg_stoi
    
    


# Training loop
for epoch in range(args.num_epochs):
    print(f"Epoch {epoch+1}/{args.num_epochs}")
    
    # Train the model
    train_loss = train_step(model= model, 
               train_dataloader = train_dataloader, 
               losses =losses, optimizer = optimizer, 
               audio_processor = audio_processor,
               compressor = compressor,
               
               metric_discriminator = metric_discriminator,
               metric_discriminator_optimizer = metric_discriminator_optimizer,
               scheduler =scheduler, 
               device = device,
               magnitude_head = args.magnitude_head, 
               phase_head= args.phase_head,complex_head= args.complex_head,                  
               epoch_num =epoch, experiment=experiment,
               )
    
    print(f"Avg. training loss: {train_loss}")

    if experiment is not None:
        experiment.log_metric("avg_train_loss", train_loss, step=epoch)
    model.eval()
    # Save the enhanced waveforms and compute metrics
    if epoch % args.compute_metrics_interval == 0:
        
       
        with torch.no_grad():
            for i in tqdm(range(len(test_dataset)), total=len(test_dataset), desc="Generating enhanced audio files"):
                filename = test_dataset.filenames[i]
                original_audio, original_sr = torchaudio.load(config["noisy_test_folder"] + filename)
                norm_factor = torch.sqrt(original_audio.shape[1] / torch.sum(original_audio ** 2))
                len_original_audio = original_audio.shape[1]

                batch = test_dataset.__getitem__(i)
                for k, v in batch.items():
                    batch[k] = v.unsqueeze(dim=0).to(device)

                input_mag = batch["input_stft"]
                input_phase = batch["input_phase"]
                
                if compressor is not None:
                    input_mag = compressor.compress(input_mag)
                    batch["input_stft"] = input_mag

                predicted_mag, predicted_phase = model(batch)

                if config["type"] == "masking" and args.magnitude_head is not None:
                    predicted_mag = predicted_mag * input_mag


                predicted_mag = predicted_mag.squeeze(0)
                predicted_phase = predicted_phase.squeeze(0)

                # ----------------- Decompress the predicted spectrogram -----------------
                if compressor is not None: predicted_mag = compressor.decompress(predicted_mag)
                else: mag_predicted = predicted_mag

                enhanced_wav = audio_processor.get_waveform_from_spectrogram(predicted_mag, predicted_phase, target_length=len_original_audio)
                enhanced_wav = enhanced_wav / norm_factor
                
                enhanced_wav = enhanced_wav.unsqueeze(dim=0)

                # Save the reconstructed waveform to a WAV file with the same name as the original file
                output_filename = os.path.splitext(os.path.basename(filename))[0] + ".wav"
                output_file = os.path.join(args.reconstructed_audio_folder, output_filename)
                torchaudio.save(output_file, enhanced_wav.to("cpu"), config["sample_rate"])
    # Open a file for storing PESQ and STOI scores
    score_file = open("pesq_stoi_scores.txt", "w")

    # Compute and log metrics after generating enhanced waveforms
    clean_dir = config["clean_test_folder"]
    enhanced_dir = args.reconstructed_audio_folder
    avg_pesq_score, avg_stoi_score = compute_metrics(clean_dir, enhanced_dir)

    if experiment is not None:
        experiment.log_metric("avg_pesq_score", avg_pesq_score, step=epoch)
        experiment.log_metric("avg_stoi_score", avg_stoi_score, step=epoch)
        
    
    # Print PESQ and STOI scores
    print(f"Avg. PESQ score: {avg_pesq_score}")
    print(f"Avg. STOI score: {avg_stoi_score}")
    # Write PESQ and STOI scores to the file
    # Open a file for storing PESQ and STOI scores
    with open("pesq_stoi_scores.txt", "a") as score_file:
        score_file.write(f"Epoch {epoch+1}\n")
        score_file.write(f"Avg. PESQ: {avg_pesq_score}\n")
        score_file.write(f"Avg. STOI: {avg_stoi_score}\n\n")

    # Save the model if needed
    save_model_if_needed(model, compressor, model_checkpoints, best_train_loss, epoch, args)

    if train_loss < best_train_loss:
        best_train_loss = train_loss
        early_stopping_counter = 0
        best_model = model.state_dict()
    else:
        early_stopping_counter += 1
        if args.adaptive_lr:
            print("Halving the learning rate.")
            for param_group in optimizer.param_groups:
                param_group["lr"] /= 2
    if early_stopping_counter >= args.patience_early_stopping:
        print("Early stopping, no improvement in train loss for {} epochs.".format(args.patience_early_stopping))
        break
# Close the score file
score_file.close()
# Save the best model
torch.save(best_model, os.path.join(args.checkpoint_dir, f"best_model.pt"))
if compressor is not None:
    print(f"Saving compressor")
    torch.save(compressor.state_dict(), os.path.join(args.checkpoint_dir, f"compressor_best_model.pt"))
    