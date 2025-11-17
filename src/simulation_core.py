# goal: generate flags to analyze in main.py

from dataclasses import dataclass
import torch
import torch.nn as nn

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