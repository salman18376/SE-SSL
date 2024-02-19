c_path="/home/salman/SE_Self-Supervise_Learning-/audio/clean_testset_wav_16k"
e_path="//home/salman/SE_Self-Supervise_Learning-/reconstructed_audio/mag_only/wavlm_transformer"
python metrics.py  --e_path $e_path --c_path $c_path 
