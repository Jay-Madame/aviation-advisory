# main.py

import sys
import pprint
from src.api_setup.api_clients import grab_articles_for_geopolitical_climate
from src.geopolitical_scoring import analyze_geopolitical_context_from
import src.simulation_core as sim_core
import src.cyber_attack_simulation as cas


def set_simulation_flags_based_on_score(tension_score: float):
    sim_core.reset_all_simulation_toggles()
    # Not of immediate concern
    if tension_score < -0.15:
        sim_core.is_trajectory_suspicious = True
        sim_core.is_radar_tracking_anomaly_detected = True
        print("    * Low-levels of Tension Detected (Score < -0.15): Trajectory and Radar flags set.")
    
    # A bit more concerning
    if tension_score < -0.3:
        sim_core.is_satellite_warning_detected = True
        sim_core.is_comms_blackout_detected = True
        print("    * Conerning Amount of Tension Detected (Score < -0.3): Satellite and Comms flags set.")

    # Very concerning
    if tension_score < -0.5:
        sim_core.is_explosion_detected = True
        sim_core.is_thermal_signature_detected = True
        print("    * CRITICAL Tension Detected (Score < -0.5): Explosion and Thermal flags set.")


def run_threat_assessment(average_tension_score: float):
    # Get the current flags that were just set by the score logic
    real_flags_snapshot = sim_core.get_current_flags_as_binary()
    print(f"    * Snapshot of Real Flags: {real_flags_snapshot}")
    
    # Define a hypothetical attack scenario: attacker zeroes out the first three flags
    attack_flags = [
        0, 0, 0, 
        real_flags_snapshot[3], real_flags_snapshot[4], real_flags_snapshot[5], real_flags_snapshot[6]
    ]
    
    print(f"    * Hypothetical Attacker Input: {attack_flags}")

    final_threat_report = cas.simulate_nuclear_scenarios(
        current_geo_score=average_tension_score,
        attack_flags=attack_flags
    )
    
    print("\n--- 🚨 FINAL THREAT ASSESSMENT REPORT (ML Analysis) ---")
    pprint.pprint(final_threat_report)


def main():
    print("\n[STEP 1] Fetching current geopolitical news articles...")
    raw_articles = grab_articles_for_geopolitical_climate()
    
    if not raw_articles and sim_core.LOADED_MODEL is None:
        print("Critical: No articles retrieved AND ML Model not loaded. Aborting analysis.")
        sys.exit(1)

    if not raw_articles:
        print("News does not report any conerns. Assuming good news for a safe score of 1.00")

    print("\n[STEP 2] Calculating average Geopolitical Tension Score...")
    average_tension_score = analyze_geopolitical_context_from(raw_articles)
    
    print(f"Final Average Tension Score: {average_tension_score:.4f}")

    # 3. SET SIMULATION STATE (The crucial linking step)
    print("\n[STEP 3] Setting C&C Sensor Flags based on Tension Score...")
    set_simulation_flags_based_on_score(average_tension_score)
    
    # 4. RUN THREAT ASSESSMENT AND CYBER SIMULATION
    print("\n[STEP 4] Running Threat Assessment and Cyber Simulation...")
    run_threat_assessment(average_tension_score)

    # 5. REPORT FINAL STATE (Optional: Show the original flags for debugging)
    print("\n[STEP 5] Generating Original Sensor State Report...")
    final_state = sim_core.generate_simulation_state()
    
    print("Current Simulation State (Set by Score)")
    for key, value in final_state.to_dict().items():
        status = "**TRUE**" if value else "False"
        print(f"  {key.replace('_', ' ').title()}: {status}")


if __name__ == "__main__":
    main()