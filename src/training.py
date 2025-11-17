# training.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from src.simulation_core import ThreatClassifier


def map_inputs_to_threat_level(geopolitical_score: float, flags_count: int) -> int:
    # High Threat: Very low geopolitical score OR multiple critical sensors tripped
    if geopolitical_score < -0.5 or flags_count >= 4:
        return 2 # High Threat
    
    # Medium Threat: Moderate geopolitical score OR at least one sensor tripped
    elif geopolitical_score < -0.2 or flags_count >= 1:
        return 1 # Medium Threat
    
    # Low Threat: Safe geopolitical score and no major flags
    else:
        return 0 # Low Threat


def generate_synthetic_data(num_samples: int) -> tuple:
    
    X_data = []
    Y_labels = []
    
    for _ in range(num_samples):
        # Sensor flags set
        flags = [random.randint(0, 1) for _ in range(7)]
        flags_count = sum(flags)
        
        # Geopolitical score set
        score = random.uniform(-1.00, 1.00)
        
        feature_vector = flags + [score]
        X_data.append(feature_vector)
        
        label = map_inputs_to_threat_level(score, flags_count)
        Y_labels.append(label)

    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_labels, dtype=torch.long) # Use Long for classification labels
    
    return X_tensor, Y_tensor


def train_threat_classifier(model: ThreatClassifier, X: torch.Tensor, Y: torch.Tensor, epochs: int = 100):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, Y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# --- Main Execution ---

if __name__ == "__main__":
    NUM_TRAINING_SAMPLES = 5000
    NUM_TRAINING_EPOCHS = 100
    
    # 1. Generate Data
    X_train, Y_train = generate_synthetic_data(NUM_TRAINING_SAMPLES)
    
    # 2. Initialize Model
    # 8 inputs: 7 flags + 1 score
    model = ThreatClassifier(input_size=8, num_classes=3)
    
    # 3. Train Model
    train_threat_classifier(model, X_train, Y_train, NUM_TRAINING_EPOCHS)
    
    # 4. Save Model State
    model_path = "threat_classifier_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining complete. Model saved to {model_path}")
    