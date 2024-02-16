import torch
from s3prl.s3prl import hub

def get_s3prl(model_name, input_waveforms, device='cuda'):#,model_config= config_path):
    # Instantiate the specified model
    model_instance  = getattr(hub, model_name)()
   
    
    # Move the model to the specified device
    model_instance = model_instance.to(device)
    
    # Move the input waveforms to the specified device
    input_waveforms = input_waveforms.to(device)
    

    # Extract hidden states using the model
    with torch.no_grad():
        model = model_instance(input_waveforms)["hidden_states"]
         
        model = model[-1] 
    

    return model



