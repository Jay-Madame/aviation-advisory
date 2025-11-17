# goal: generate flags to analyze in main.py

from dataclasses import dataclass
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

is_explosion_detected = False
is_seismic_anomaly_detected = False
is_radar_tracking_anomaly_detected = False
is_satellite_warning_detected = False
is_comms_blackout_detected = False
is_thermal_signature_detected = False
is_trajectory_suspicious = False

@dataclass
class SimulationState:
    explosion_detected: bool
    seismic_anomaly_detected: bool
    radar_tracking_anomaly_detected: bool
    satellite_warning_detected: bool
    comms_blackout_detected: bool
    thermal_signature_detected: bool
    trajectory_suspicious: bool

    def to_dict(self) -> dict:
        return {
            "explosion_detected": self.explosion_detected,
            "seismic_anomaly_detected": self.seismic_anomaly_detected,
            "radar_tracking_anomaly_detected": self.radar_tracking_anomaly_detected,
            "satellite_warning_detected": self.satellite_warning_detected,
            "comms_blackout_detected": self.comms_blackout_detected,
            "thermal_signature_detected": self.thermal_signature_detected,
            "trajectory_suspicious": self.trajectory_suspicious
        }

def generate_simulation_state() -> SimulationState:
    return SimulationState(
        explosion_detected=is_explosion_detected,
        seismic_anomaly_detected=is_seismic_anomaly_detected,
        radar_tracking_anomaly_detected=is_radar_tracking_anomaly_detected,
        satellite_warning_detected=is_satellite_warning_detected,
        comms_blackout_detected=is_comms_blackout_detected,
        thermal_signature_detected=is_thermal_signature_detected,
        trajectory_suspicious=is_trajectory_suspicious
    )

def set_flags_manually(binary_input: list):
    if len(binary_input) != 7:
        raise ValueError("Input must be in binary form to contain 7 integers (0 or 1 values)")

    global is_explosion_detected
    global is_seismic_anomaly_detected
    global is_radar_tracking_anomaly_detected
    global is_satellite_warning_detected
    global is_comms_blackout_detected
    global is_thermal_signature_detected
    global is_trajectory_suspicious
    
    is_explosion_detected = bool(binary_input[0])
    is_seismic_anomaly_detected = bool(binary_input[1])
    is_radar_tracking_anomaly_detected = bool(binary_input[2])
    is_satellite_warning_detected = bool(binary_input[3])
    is_comms_blackout_detected = bool(binary_input[4])
    is_thermal_signature_detected = bool(binary_input[5])
    is_trajectory_suspicious = bool(binary_input[6])

def reset_all_simulation_toggles():
    """
    Reset all toggles to their default 'False' state.
    Useful for debugging or simulation resets.
    """
    global is_explosion_detected
    global is_seismic_anomaly_detected
    global is_radar_tracking_anomaly_detected
    global is_satellite_warning_detected
    global is_comms_blackout_detected
    global is_thermal_signature_detected
    global is_trajectory_suspicious

    is_explosion_detected = False
    is_seismic_anomaly_detected = False
    is_radar_tracking_anomaly_detected = False
    is_satellite_warning_detected = False
    is_comms_blackout_detected = False
    is_thermal_signature_detected = False
    is_trajectory_suspicious = False


def debug_print_state():
    state = generate_simulation_state()
    print("Current Simulation State:")
    for key, value in state.to_dict().items():
        print(f"  {key}: {value}")

class ThreatClassifier(nn.Module):
    def __init__(self, input_size=8, hidden_size=16, num_classes=3):
        super(ThreatClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(hidden_size, num_classes) 

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define the expected path to the saved model file
MODEL_FILE_NAME = "threat_classifier_model.pth"
MODEL_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', MODEL_FILE_NAME)

def get_current_flags_as_binary() -> list:
    state = generate_simulation_state().to_dict()
    flag_keys = [
        "explosion_detected", "seismic_anomaly_detected", 
        "radar_tracking_anomaly_detected", "satellite_warning_detected", 
        "comms_blackout_detected", "thermal_signature_detected", 
        "trajectory_suspicious"
    ]
    return [1 if state[key] else 0 for key in flag_keys]

def load_trained_model(path: str = MODEL_FILE_PATH) -> ThreatClassifier:
    """
    Initializes the model and loads the trained weights from the specified path.
    Returns None if the file is not found.
    """
    model = ThreatClassifier(input_size=8, num_classes=3)
    try:
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
    except FileNotFoundError:
        print(f"WARNING: Trained model not found at {path}. Threat prediction unavailable.")
        return None
    except Exception as e:
        print(f"ERROR: Failed to load model weights: {e}")
        return None

# Load the model once when the module is imported
LOADED_MODEL = load_trained_model()


def predict_threat_level(binary_flags: list, geo_score: float, model: ThreatClassifier = LOADED_MODEL) -> int:
    """
    Predicts the threat class (0, 1, or 2) using the loaded PyTorch model.
    """
    if model is None:
        return -1 # Indicates prediction failure
    
    # Prepare input tensor: [7 flags, 1 score]
    flags_tensor = torch.tensor(binary_flags, dtype=torch.float32)
    score_tensor = torch.tensor([geo_score], dtype=torch.float32)
    input_vector = torch.cat((flags_tensor, score_tensor)).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        output = model(input_vector)
        # Convert logits to probability distribution
        probabilities = F.softmax(output, dim=1)
        # Get the class with the highest probability
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
    return predicted_class