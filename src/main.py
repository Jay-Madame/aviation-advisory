# main.py

from src.api_setup.api_clients import grab_articles_for_geopolitical_climate
from src.geopolitical_scoring import analyze_geopolitical_context_from
import src.simulation_core as sim_core # Import the module to access toggles and functions


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

def main():
    print("\n[STEP 1] Fetching current geopolitical news articles...")
    raw_articles = grab_articles_for_geopolitical_climate()
    
    if not raw_articles:
        print("News does not report any conerns. Assuming good news for a safe score of 1.00")

    print("\n[STEP 2] Calculating average Geopolitical Tension Score...")
    average_tension_score = analyze_geopolitical_context_from(raw_articles)
    
    print(f"Final Average Tension Score: {average_tension_score:.4f}")

    # 3. SET SIMULATION STATE (The crucial linking step)
    print("\n[STEP 3] Setting C&C Sensor Flags based on Tension Score...")
    set_simulation_flags_based_on_score(average_tension_score)
    
    # 4. REPORT FINAL STATE
    print("\n[STEP 4] Generating Final Sensor State Report...")
    final_state = sim_core.generate_simulation_state()
    
    print("Current Simulation State")
    for key, value in final_state.to_dict().items():
        status = "**TRUE**" if value else "False"
        print(f"  {key.replace('_', ' ').title()}: {status}")



if __name__ == "__main__":
    main()