"""
V6 Clinical Scenario-Based Prompts for Sepsis Treatment

Key innovation over v5:
- Foresight based on CURRENT PHASE (treatment window + clinical scenario)
- Orthogonal to Task (baseline profile) and Hindsight (dynamic changes)
- Information sources: current vitals, cumulative treatment response, phase-specific goals

Design:
- Task: Baseline patient profile (unchanged from v4/v5)
- Hindsight: Dynamic response (unchanged from v4/v5)
- Foresight: Treatment window + Clinical scenario + Time-bound goals + Constraints

Expected outcomes:
- Task-Foresight CCA < 0.5 (vs v5's 0.88)
- 1728+ Foresight variants (4 windows × 6 scenarios × 12 goals × 6 constraints)
"""

import numpy as np
from collections import defaultdict

# =============================================================================
# Part 1: Classification Functions (same as v4/v5)
# =============================================================================

def _classify_age(age):
    """Classify age into categories."""
    if age < 50:
        return 'young'
    elif age < 70:
        return 'middle-aged'
    else:
        return 'elderly'


def _classify_comorbidities(state):
    """Extract comorbidities from state vector."""
    comorbidities = []
    creatinine_base = state[22]
    platelets_base = state[16]
    pfr_base = state[27]

    if creatinine_base > 2.0:
        comorbidities.append("CKD")
    if platelets_base < 100:
        comorbidities.append("Thrombocytopenia")
    if pfr_base < 300:
        comorbidities.append("Chronic_Lung_Disease")

    return comorbidities if comorbidities else ["None"]


def _classify_infection_source(state):
    """Classify probable infection source."""
    admission_type = int(state[42]) if len(state) > 42 else 0

    if admission_type == 0:
        return "Unknown"
    elif admission_type == 1:
        return "Pulmonary"
    elif admission_type == 2:
        return "Abdominal"
    elif admission_type == 3:
        return "Urinary"
    else:
        return "Other"


def _classify_shock_type(state):
    """Classify shock type from physiological parameters."""
    map_val = state[2]
    lactate = state[24] if len(state) > 24 else 0

    if map_val < 65 and lactate >= 4:
        return "Septic_Shock"
    elif map_val < 70:
        return "Shock"
    else:
        return "Sepsis"


def _classify_dominant_organs(acuities):
    """Identify dominant organ dysfunction from SAPS2 components."""
    organ_names = ['CNS', 'CVS', 'Renal', 'Resp', 'Hepatic', 'Coag']
    scores = acuities[3:9]

    dominant = [name for name, score in zip(organ_names, scores) if score > 0]
    return dominant if dominant else ["None"]


def _classify_fluid_response(curr_state, prev_state, action_id):
    """Classify fluid resuscitation response based on action ID."""
    fluid_delta = curr_state[39] - prev_state[39] if len(curr_state) > 39 else 0
    map_delta = curr_state[2] - prev_state[2]

    if action_id == 0:  # Aggressive IV fluid
        if fluid_delta > 500 and map_delta >= 10:
            return "Excellent"
        elif fluid_delta > 500 and map_delta >= 5:
            return "Good"
        elif fluid_delta > 500:
            return "Partial"
        else:
            return "Poor"
    elif action_id == 1:  # Conservative IV fluid
        return "Conservative"
    elif action_id == 16:  # Diuretic
        return "Diuresis"
    else:
        return "None"


def _classify_vaso_response(curr_state, prev_state, action_id):
    """Classify vasopressor response based on action ID."""
    map_delta = curr_state[2] - prev_state[2]
    hr_delta = curr_state[1] - prev_state[1]

    if action_id in [2, 3]:  # Vaso active
        if map_delta >= 5 and abs(hr_delta) < 10:
            return "Stable"
        elif map_delta < 0:
            return "Worsening"
        elif map_delta >= 5:
            return "Improved"
        else:
            return "Partial"
    elif action_id == 4:  # Vaso down
        return "Weaned"
    else:
        return "None"


# =============================================================================
# Part 2: V6 New Functions - Treatment Window & Clinical Scenario
# =============================================================================

def _classify_treatment_window(state, acuities, t, cumulative_fluid):
    """
    Classify current treatment window based on hemodynamic status and phase.

    6 windows (V6.1: increased granularity):
    - Emergent_Resuscitation: MAP < -1.18 (10th percentile) 或 Lactate > 1.5
    - Active_Resuscitation: MAP < -0.74 (25th percentile) 或 Lactate > 1.0
    - Early_Stabilization: MAP < -0.10 (50th percentile), t<6
    - Late_Stabilization: MAP < -0.10 (50th percentile), t>=6
    - Optimization: MAP >= 0.67 (75th percentile) 且 Lactate < 0.5
    - De_escalation: MAP >= 0.67 且 Lactate < 0.5 且 t>=12

    注意：使用 z-score 阈值（数据已标准化），而不是原始临床阈值

    Args:
        state: current state vector (45,) - standardized z-scores
        acuities: SAPS2 acuities (10,)
        t: current timestep
        cumulative_fluid: cumulative fluid balance (L, from state[39])

    Returns:
        window: str, one of 6 treatment windows
    """
    map_val = state[2]  # z-score
    lactate = state[24] if len(state) > 24 else 0  # z-score

    # Emergent Resuscitation: severe shock (bottom 10%)
    if map_val < -1.18 or lactate > 1.5:
        return "Emergent_Resuscitation"

    # Active Resuscitation: moderate shock (bottom 25%)
    if map_val < -0.74 or lactate > 1.0:
        return "Active_Resuscitation"

    # De-escalation: stable for extended period (top 25%)
    if map_val >= 0.67 and lactate < 0.5 and t >= 12:
        return "De_escalation"

    # Optimization: fully stable (top 25%)
    if map_val >= 0.67 and lactate < 0.5:
        return "Optimization"

    # Early vs Late Stabilization (below median)
    if t < 6:
        return "Early_Stabilization"
    else:
        return "Late_Stabilization"


def _classify_clinical_scenario(state, acuities):
    """
    Classify clinical scenario based on current organ dysfunction combination.

    12 scenarios (V6.1: increased granularity):

    Single organ (6):
    - Isolated_Shock: MAP < -0.74 (25th percentile), 其他器官正常
    - Isolated_AKI: Creatinine > 0.5 (top 31%), 其他器官正常
    - Isolated_ARDS: PFR < -0.5, 其他器官正常
    - Isolated_CNS: GCS < -1.5, 其他器官正常
    - Isolated_Coagulopathy: Platelets < -1.0, 其他器官正常
    - No_Organ_Failure: 所有器官正常

    Double organ (4):
    - Shock_with_AKI: MAP < -0.74 + Creatinine > 0.5
    - Shock_with_ARDS: MAP < -0.74 + PFR < -0.5
    - AKI_with_ARDS: Creatinine > 0.5 + PFR < -0.5
    - Shock_with_CNS: MAP < -0.74 + GCS < -1.5

    Multi-organ (2):
    - Triple_organ_Failure: ≥3 个器官功能障碍
    - Multi_organ_Dysfunction: 2 个器官功能障碍（非上述组合）

    注意：使用 z-score 阈值（数据已标准化），而不是原始临床阈值

    Returns:
        scenario: str, one of 12 clinical scenarios
    """
    map_val = state[2]  # z-score
    lactate = state[24] if len(state) > 24 else 0  # z-score
    creatinine = state[22] if len(state) > 22 else 0  # z-score
    pfr = state[27] if len(state) > 27 else 999  # z-score
    gcs = state[0]  # z-score
    platelets = state[16] if len(state) > 16 else 200  # z-score
    bilirubin = state[34] if len(state) > 34 else 0  # z-score

    # Count organ failures (using z-score thresholds)
    organ_failures = []
    if map_val < -0.74:  # bottom 25%
        organ_failures.append("CVS")
    if creatinine > 0.5:  # ~top 31% (z > 0.5)
        organ_failures.append("Renal")
    if pfr < -0.5:  # ~bottom 31%
        organ_failures.append("Resp")
    if gcs < -1.5:  # severe
        organ_failures.append("CNS")
    if platelets < -1.0:  # ~bottom 16%
        organ_failures.append("Coag")
    if bilirubin > 1.0:  # ~top 16%
        organ_failures.append("Hepatic")

    # No organ failure
    if len(organ_failures) == 0:
        return "No_Organ_Failure"

    # Triple organ failure
    if len(organ_failures) >= 3:
        return "Triple_organ_Failure"

    # Single organ scenarios
    if len(organ_failures) == 1:
        if "CVS" in organ_failures:
            return "Isolated_Shock"
        elif "Renal" in organ_failures:
            return "Isolated_AKI"
        elif "Resp" in organ_failures:
            return "Isolated_ARDS"
        elif "CNS" in organ_failures:
            return "Isolated_CNS"
        elif "Coag" in organ_failures:
            return "Isolated_Coagulopathy"
        else:
            return f"Isolated_{organ_failures[0]}"

    # Double organ scenarios (len == 2)
    if "CVS" in organ_failures and "Renal" in organ_failures:
        return "Shock_with_AKI"
    elif "CVS" in organ_failures and "Resp" in organ_failures:
        return "Shock_with_ARDS"
    elif "Renal" in organ_failures and "Resp" in organ_failures:
        return "AKI_with_ARDS"
    elif "CVS" in organ_failures and "CNS" in organ_failures:
        return "Shock_with_CNS"
    else:
        return "Multi_organ_Dysfunction"


def _compute_time_bound_goals(window, state, acuities, t):
    """
    Compute time-bound goals based on treatment window (V6.1: increased specificity).

    Returns:
        goals: dict with keys '3h', '6h', '24h'
    """
    map_val = state[2]
    lactate = state[24] if len(state) > 24 else 0
    creatinine = state[22] if len(state) > 22 else 0

    goals = {}

    if window == "Emergent_Resuscitation":
        goals['3h'] = "MAP>60_Lac↓20%"
        goals['6h'] = "MAP>65_Lac<4"
        goals['24h'] = "Lac<2_Source_control"

    elif window == "Active_Resuscitation":
        goals['3h'] = "MAP>65_Lac↓10%"
        goals['6h'] = "Lac<2"
        goals['24h'] = "Organ_stable"

    elif window == "Early_Stabilization":
        goals['3h'] = "MAP≥70"
        goals['6h'] = "Lac<1.5"
        goals['24h'] = "No_new_failure"

    elif window == "Late_Stabilization":
        goals['3h'] = "Maintain_MAP≥70"
        goals['6h'] = "Lac<1"
        goals['24h'] = "De-escalate_vaso"

    elif window == "Optimization":
        goals['3h'] = "Maintain_euvolemia"
        goals['6h'] = "Prevent_complications"
        goals['24h'] = "De-escalate_ABX"

    elif window == "De_escalation":
        goals['3h'] = "Maintain_stable"
        goals['6h'] = "Stop_vaso_wean_fluids"
        goals['24h'] = "ICU_discharge_ready"

    else:
        goals['3h'] = "Stabilize"
        goals['6h'] = "Improve"
        goals['24h'] = "Recover"

    return goals


def _get_treatment_constraints(state, acuities, cumulative_fluid, vaso_duration):
    """
    Get treatment constraints based on current state and cumulative treatment.

    Args:
        state: current state vector (45,)
        acuities: SAPS2 acuities (10,)
        cumulative_fluid: cumulative fluid balance (L)
        vaso_duration: duration of vasopressor use (hours)

    Returns:
        constraints: list of constraint strings
    """
    constraints = []

    map_val = state[2]
    lactate = state[24] if len(state) > 24 else 0
    creatinine = state[22] if len(state) > 22 else 0
    pfr = state[27] if len(state) > 27 else 999
    gcs = state[0]

    # Fluid-related constraints
    if cumulative_fluid > 5 and map_val >= 65:
        constraints.append("Avoid_fluid_overload")
    elif map_val < 65 and cumulative_fluid > 3:
        constraints.append("Maintain_euvolemia")

    # Vasopressor-related constraints
    if vaso_duration > 48:
        constraints.append("Limit_vaso_duration")

    # Organ-specific constraints
    if pfr < 200:
        constraints.append("Lung_protective")
    if creatinine >= 2:
        constraints.append("Renal_protect")
    if gcs < 8:
        constraints.append("Neuro_protect")

    return constraints if constraints else ["Standard_care"]


# =============================================================================
# Part 3: Template Builders (Task/Hindsight same as v4/v5)
# =============================================================================

def _build_task_text(state, acuities):
    """Build TASK embedding: static patient profile (same as v4/v5)."""
    age = state[43] if len(state) > 43 else 65
    gender = "F" if (len(state) > 44 and state[44] > 0.5) else "M"

    age_cat = _classify_age(age)
    comorbidities = _classify_comorbidities(state)
    source = _classify_infection_source(state)
    shock = _classify_shock_type(state)
    organs = _classify_dominant_organs(acuities)

    parts = [
        f"Patient: {age}y {gender} | Age: {age_cat}",
        f"PMH: {','.join(comorbidities)} | PX: {source}",
        f"Pathophysiology: {shock}",
        f"Burden: {','.join(organs)}",
    ]

    return " ".join(parts)


def _build_hindsight_text(curr_state, prev_state, curr_action_id, prev_action_id=None):
    """Build HINDSIGHT embedding: dynamic changes (same as v4/v5)."""
    hr_delta = curr_state[1] - prev_state[1]
    map_delta = curr_state[2] - prev_state[2]
    lactate_delta = (curr_state[24] - prev_state[24]) if len(curr_state) > 24 else 0

    hr = int(curr_state[1])
    map_val = int(curr_state[2])
    sao2 = int(curr_state[5]) if len(curr_state) > 5 else 98

    lactate = curr_state[24] if len(curr_state) > 24 else 0
    creatinine = curr_state[22] if len(curr_state) > 22 else 0
    platelets = int(curr_state[16]) if len(curr_state) > 16 else 200

    fluid_resp = _classify_fluid_response(curr_state, prev_state, curr_action_id)
    vaso_resp = _classify_vaso_response(curr_state, prev_state, curr_action_id)

    hr_delta_str = f"{hr_delta:+.0f}" if abs(hr_delta) > 0 else "0"
    map_delta_str = f"{map_delta:+.0f}" if abs(map_delta) > 0 else "0"
    lac_delta_str = f"{lactate_delta:+.1f}" if abs(lactate_delta) > 0.1 else "0"

    parts = [
        f"Δ1h: HR{hr_delta_str} MAP{map_delta_str} LAC{lac_delta_str}",
        f"Current: HR={hr} MAP={map_val} SpO2={sao2}",
        f"Labs: Lac={lactate:.1f} Cr={creatinine:.1f} Plt={platelets}",
        f"Response: Fluid→{fluid_resp} Vaso→{vaso_resp}",
    ]

    return " ".join(parts)


# =============================================================================
# Part 4: V6 Foresight Builder - Clinical Scenario-Based
# =============================================================================

def _build_foresight_text(curr_state, curr_acuities, t, cumulative_fluid, vaso_duration):
    """
    Build FORESIGHT embedding: V6 clinical scenario-based version.

    Key features:
    - Treatment window (Resuscitation/Stabilization/Optimization/De-escalation)
    - Clinical scenario (6 combinations of organ dysfunctions)
    - Time-bound goals (3h/6h/24h)
    - Treatment constraints (based on cumulative treatment response)

    Args:
        curr_state: current state vector (45,)
        curr_acuities: SAPS2 acuities (10,)
        t: current timestep
        cumulative_fluid: cumulative fluid balance (L)
        vaso_duration: duration of vasopressor use (hours)

    Returns:
        foresight_text: str
    """
    # Classify treatment window
    window = _classify_treatment_window(curr_state, curr_acuities, t, cumulative_fluid)

    # Classify clinical scenario
    scenario = _classify_clinical_scenario(curr_state, curr_acuities)

    # Compute time-bound goals
    goals = _compute_time_bound_goals(window, curr_state, curr_acuities, t)

    # Get treatment constraints
    constraints = _get_treatment_constraints(curr_state, curr_acuities, cumulative_fluid, vaso_duration)

    # Build structured text
    parts = [
        f"Window:{window}",
        f"Scenario:{scenario}",
        f"3h:{goals['3h']}",
        f"6h:{goals['6h']}",
        f"24h:{goals['24h']}",
        f"Constraints:{','.join(constraints[:3])}",  # Limit to 3 constraints
    ]

    return "|".join(parts)


# =============================================================================
# Part 5: Main Interface
# =============================================================================

def build_prompt_sequences_for_trajectory(states, acuities, rtgs, actions, max_timestep=20):
    """
    Build prompt sequences for a trajectory (V6.1: use real cumulative state).

    Compatible with v2/v3/v4/v5 interface.

    Args:
        states: (T, 45) array of demographic state vectors
        acuities: (T, 10) array of SAPS2 component scores
        rtgs: (T,) array of rewards-to-go (unused in v6)
        actions: (T,) array of discrete action IDs (int64)
        max_timestep: maximum trajectory length

    Returns:
        dict with keys:
            - task_prompts: list of T task strings
            - hindsight_prompts: list of T hindsight strings
            - foresight_prompts: list of T foresight strings
    """
    T = min(len(states), max_timestep)

    task_prompts = []
    hindsight_prompts = []
    foresight_prompts = []

    for t in range(T):
        curr_state = states[t]
        curr_acuities = acuities[t]

        # Task: use initial state for all timesteps (time-invariant)
        if t == 0:
            task_text = _build_task_text(states[0], acuities[0])
        else:
            task_text = task_prompts[0]  # Reuse

        # Hindsight: current state vs previous state
        if t > 0:
            prev_state = states[t-1]
            curr_action_id = int(actions[t]) if t < len(actions) else 0
            prev_action_id = int(actions[t-1]) if t > 0 else 0
            hindsight_text = _build_hindsight_text(curr_state, prev_state, curr_action_id, prev_action_id)
        else:
            # At t=0, no previous state
            hr = int(curr_state[1])
            map_val = int(curr_state[2])
            sao2 = int(curr_state[5]) if len(curr_state) > 5 else 98
            lactate = curr_state[24] if len(curr_state) > 24 else 0
            creatinine = curr_state[22] if len(curr_state) > 22 else 0
            platelets = int(curr_state[16]) if len(curr_state) > 16 else 200
            hindsight_text = f"Δ1h: None Current: HR={hr} MAP={map_val} SpO2={sao2} Labs: Lac={lactate:.1f} Cr={creatinine:.1f} Plt={platelets} Response: Initial"

        # Foresight: V6.1 clinical scenario-based (use REAL cumulative state)
        # Get cumulative fluid balance from state[39] (in L)
        cumulative_fluid = curr_state[39] if len(curr_state) > 39 else 0.0

        # Estimate vaso duration based on MAP and time
        # If MAP < 70, likely on vasopressors
        map_val = curr_state[2]
        if map_val < 70:
            # Rough estimate: vaso duration increases with t if MAP remains low
            vaso_duration = t  # Simplified: assume on vaso since t=0 if MAP<70
        else:
            # MAP >= 70: estimate vaso duration based on recent history
            vaso_duration = 0
            if t > 0:
                # Check if recently had low MAP
                for prev_t in range(max(0, t-5), t):
                    if states[prev_t][2] < 70:
                        vaso_duration = max(vaso_duration, t - prev_t)

        foresight_text = _build_foresight_text(
            curr_state, curr_acuities, t, cumulative_fluid, vaso_duration
        )

        task_prompts.append(task_text)
        hindsight_prompts.append(hindsight_text)
        foresight_prompts.append(foresight_text)

    return {
        "task_prompts": task_prompts,
        "hindsight_prompts": hindsight_prompts,
        "foresight_prompts": foresight_prompts,
    }


# =============================================================================
# Part 6: Testing/Preview Functions
# =============================================================================

if __name__ == "__main__":
    # Test with dummy data
    print("=== V6.1 Clinical Scenario Prompts Test ===\n")

    # Create dummy trajectory with varying severity
    T = 10
    base_state = [15, 90, 60, 3.5, 38, 98, 7.4, 120, 36, 0.7,
                 1.5, 140, 4, 7, 180, 22, 2.5, 0, 0, 0,
                 0, 0, 1.0, 250, 4.0, 0, 0, 0, 250, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 2.0, 1, 65, 1]  # state[39]=2.0L cumulative fluid
    states = np.array([base_state] * T)

    # Simulate improvement over time
    for t in range(T):
        improvement_factor = t * 0.1
        states[t, 2] = min(85, 60 + improvement_factor * 25)  # MAP improves
        states[t, 24] = max(0.8, 4.0 - improvement_factor * 3.2)  # Lactate improves
        states[t, 22] = max(1.0, 2.5 - improvement_factor * 1.5)  # Creatinine improves
        states[t, 39] = 2.0 + t * 0.3  # Cumulative fluid increases

    acuities = np.array([
        [0, 0, 0, 0, 6, 6, 0, 0, 0, 0],
    ] * T).reshape(T, 10)

    rtgs = np.zeros(T)
    actions = np.array([0, 0, 2, 3, 4, 1, 15, 15, 15, 15])  # Aggressive fluid → vaso → wean → stable

    # Build prompts
    sequences = build_prompt_sequences_for_trajectory(
        states=states,
        acuities=acuities,
        rtgs=rtgs,
        actions=actions,
        max_timestep=20,
    )

    # Print all timesteps to show evolution
    for t in range(T):
        print(f"--- Timestep {t} ---")
        print(f"Task:      {sequences['task_prompts'][t]}")
        print(f"Hindsight: {sequences['hindsight_prompts'][t]}")
        print(f"Foresight: {sequences['foresight_prompts'][t]}")
        print()

    print("=== Test Complete ===")
    print("V6.1 Key Features:")
    print("- 6 Treatment windows (vs V6's 4)")
    print("- 12 Clinical scenarios (vs V6's 6)")
    print("- Time-bound goals with increased specificity")
    print("- Treatment constraints (based on REAL cumulative fluid from state)")
    print("- Foresight information orthogonal to Task/Hindsight")
    print("")
    print("V6.1 Improvements over V6:")
    print("- Treatment windows: 4 → 6 (Emergent/Active Resuscitation)")
    print("- Clinical scenarios: 6 → 12 (Isolated/Multi-organ细分)")
    print("- Cumulative state: action-based → REAL state[39] fluid balance")
    print("- Expected: Foresight norm_std >> 0.16 (V6's value)")
