# src/cyber_attack_simulation.py

import pprint
from typing import List
from src.simulation_core import LOADED_MODEL, predict_threat_level
from src.simulation_core import set_flags_manually, reset_all_simulation_toggles
from src.simulation_core import get_current_flags_as_binary

# Helper map to translate the ML model's output (0, 1, 2) into readable threat levels
THREAT_MAP = {
    0: "LOW RISK (All Clear)", 
    1: "MEDIUM THREAT (Warning)", 
    2: "HIGH THREAT (Attack Imminent)",
    -1: "PREDICTION FAILED (Model Error)"
}


def simulate_nuclear_scenarios(current_geo_score: float, attack_flags: List[int]) -> dict:
    """
    Compares the ML threat prediction based on the ACTUAL sensor state 
    vs. the state AFTER a hypothetical cyber attack manipulation.
    """
    
    if LOADED_MODEL is None:
        return {"result": "Simulation Failed: ML Model is not available."}

    # 1. BASELINE SCENARIO (What the system ACTUALLY sees)
    real_flags = get_current_flags_as_binary()
    real_prediction_class = predict_threat_level(real_flags, current_geo_score, LOADED_MODEL)
    
    # 2. CYBER ATTACK SCENARIO (What the system is fed by the attacker)
    attack_prediction_class = predict_threat_level(attack_flags, current_geo_score, LOADED_MODEL)

    # 3. ANALYSIS
    attack_succeeded = real_prediction_class != attack_prediction_class
    
    return {
        "geopolitical_score": current_geo_score,
        "real_sensor_state": real_flags,
        "real_threat_ml": THREAT_MAP.get(real_prediction_class),
        "attack_sensor_state": attack_flags,
        "attack_threat_ml": THREAT_MAP.get(attack_prediction_class),
        "attack_succeeded": attack_succeeded,
        "summary": (
            "Attack Successfully MASKED the threat." if attack_succeeded and attack_prediction_class < real_prediction_class else
            "Attack Successfully FORCED a false alert." if attack_succeeded and attack_prediction_class > real_prediction_class else
            "Attack FAILED to change the final ML classification."
        )
    }


if __name__ == '__main__':
    
    # SCENARIO: MASKING A REAL ATTACK (Cyber Sabotage)
    real_geo_score = -0.90
    set_flags_manually([1, 0, 1, 1, 0, 0, 1]) 
    hacker_flags = [0, 0, 0, 0, 0, 0, 0] 
    
    print("--- SCENARIO: MASKING A REAL ATTACK (Cyber Sabotage) ---")
    report_a = simulate_nuclear_scenarios(real_geo_score, hacker_flags)
    pprint.pprint(report_a)

    print("\n--- SCENARIO: FORCING A FALSE ALARM (Cyber Escalation) ---")
    # SCENARIO: FORCING A FALSE ALARM (Cyber Escalation)
    real_geo_score = 0.85
    set_flags_manually([0, 0, 0, 0, 0, 0, 0]) 
    hacker_flags_b = [1, 1, 1, 1, 1, 1, 1] 

    report_b = simulate_nuclear_scenarios(real_geo_score, hacker_flags_b)
    pprint.pprint(report_b)
    
    reset_all_simulation_toggles()
