import torch
import torch.nn as nn
import torch.functional as F

class Wav2Letter(nn.Module):
    def __init__(self, num_classes : int = 24, input_type : str = "wavform", num_features: int = 1):
        super().__init__()
        
        acoustic_num_features = 250 if input_type == "waveform" else num_features
        acoustic_model = nn.Sequential(
            nn.Conv1d(in_channels=acoustic_num_features, out_channels=250, kernel_size=48, stride=2, padding=23),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=2000, kernel_size=32, stride=1, padding=16),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=2000, out_channels=2000, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=2000, out_channels=num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        
        if input_type == "waveform":
            waveform_model = nn.Sequential(
                nn.Conv1d(in_channels=num_features, out_channels=250, kernel_size=250, stride=160, padding=45),
                nn.ReLU(inplace=True),
            )
            self.acoustic_model = nn.Sequential(waveform_model, acoustic_model)
            
        if input_type in ["power?spectrum", "mfcc"]:
            self.acoustic_model = acoustic_model
            
    def forward(self, x: torch.tensor):
        x = self.acoustic_model(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x
        