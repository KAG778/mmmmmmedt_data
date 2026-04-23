"""
v3 Qualitative Clinical Prompt Builder for CSP-DT.

Key design principles (vs v2):
  1. THREE embeddings encode INDEPENDENT information — no overlap
  2. Task: patient profile + treatment goal (minimal numbers, acts as anchor)
  3. History: qualitative description of past action effects + state trends (NO numbers)
  4. Foresight: qualitative risk assessment + organ protection focus (NO numbers)
  5. Conditional template selection based on clinical thresholds (like Simbid)
  6. Multiple template variants per category for diversity

This eliminates the v2 redundancy problem where all three embeddings
encoded the same SAPS2/burden/RTG information (cosine sim > 0.93).
"""
import random
import numpy as np
from typing import Dict, List, Optional, Sequence

# Action cluster descriptions (from MIMIC-III treatment clustering)
ACTION_DESCRIPTIONS = {
    0:  "aggressive IV fluid resuscitation",
    1:  "conservative IV fluid",
    2:  "vasopressor initiation",
    3:  "vasopressor titration up",
    4:  "vasopressor titration down",
    5:  "mechanical ventilation with high PEEP",
    6:  "lung-protective ventilation",
    7:  "extubation readiness assessment",
    8:  "empiric broad-spectrum antibiotics",
    9:  "antibiotic de-escalation",
    10: "renal replacement therapy consideration",
    11: "blood transfusion for anemia",
    12: "glucose control with insulin",
    13: "sedation adjustment",
    14: "anticoagulation management",
    15: "supportive care monitoring",
    16: "diuretic therapy for fluid overload",
    17: "bicarbonate for acidosis",
    18: "calcium supplementation",
    19: "nutritional support initiation",
    20: "DVT prophylaxis",
    21: "stress ulcer prophylaxis",
    22: "temperature management",
    23: "electrolyte repletion",
    24: "palliative care discussion",
}

# Organ system names for burden description
ORGAN_NAMES = [
    "cardiovascular", "respiratory", "neurological", "renal",
    "hepatic", "hematologic", "metabolic",
]

# --- Classification helpers ---

def _classify_hr(hr):
    if hr < 60:
        return 'bradycardic'
    elif hr > 100:
        return 'tachycardic'
    else:
        return 'normal'

def _classify_bp(sbp):
    if sbp < 90:
        return 'hypotensive'
    elif sbp > 180:
        return 'hypertensive'
    else:
        return 'stable'

def _classify_rr(rr):
    if rr > 24:
        return 'tachypneic'
    elif rr < 10:
        return 'bradypneic'
    else:
        return 'normal'

def _classify_temp(temp):
    if temp > 38.5:
        return 'febrile'
    elif temp < 36.0:
        return 'hypothermic'
    else:
        return 'normal'

def _classify_gcs(gcs):
    if gcs < 8:
        return 'severely impaired'
    elif gcs < 13:
        return 'impaired'
    else:
        return 'normal'

def _classify_shock_index(si):
    return 'elevated' if si >= 1.0 else 'normal'

def _classify_pfr(pfr):
    return 'impaired' if pfr < 300 else 'adequate'

def _classify_fluid(balance):
    if balance > 500:
        return 'positive'
    elif balance < -500:
        return 'negative'
    else:
        return 'neutral'

def _classify_trend(delta, threshold=0.3):
    if delta > threshold:
        return 'rising'
    elif delta < -threshold:
        return 'falling'
    else:
        return 'stable'

def _classify_lactate_level(lactate):
    if lactate >= 4.0:
        return 'critically high'
    elif lactate >= 2.0:
        return 'elevated'
    else:
        return 'normal'

def _classify_platelets(plt):
    if plt < 100:
        return 'critically low'
    elif plt < 150:
        return 'low'
    else:
        return 'normal'

def _classify_bilirubin(bili):
    return 'elevated' if bili >= 1.2 else 'normal'

def _classify_inr(inr):
    return 'elevated' if inr >= 1.5 else 'normal'

def _classify_creatinine(creatinine):
    if creatinine >= 2.0:
        return 'elevated'
    elif creatinine >= 1.2:
        return 'mildly elevated'
    else:
        return 'normal'

def _classify_wbc(wbc):
    if wbc > 12.0:
        
        
        return 'elevated'
    elif wbc < 4.0:
        return 'low'
    else:
        return 'normal'

def _classify_risk(n_abnormal):
    if n_abnormal >= 6:
        return 'critical'
    elif n_abnormal >= 4:
        return 'high'
    elif n_abnormal >= 2:
        return 'moderate'
    else:
        return 'low'


# --- Template variants (3-5 per category) ---

# Task templates
TASK_TEMPLATES = {
    'critical': [
        "CRITICAL sepsis patient requiring immediate intervention.",
        "Life-threatening sepsis, emergency treatment needed.",
        "Severe sepsis with multiple organ dysfunction, critical priority.",
    ],
    'high': [
        "Severe sepsis patient with significant organ involvement.",
        "High-acuity sepsis requiring close monitoring and active treatment.",
        "Serious sepsis presentation with organ dysfunction concerns.",
    ],
    'moderate': [
        "Sepsis patient with moderate illness severity.",
        "Moderate-risk sepsis requiring guided treatment and reassessment.",
        "Sepsis of moderate severity, responding to current management.",
    ],
    'low': [
        "Sepsis patient with mild presentation, improving trajectory.",
        "Low-severity sepsis, stable on current management.",
        "Mild sepsis, minimal organ dysfunction detected.",
    ],
}

BURDEN_TEMPLATES = {
    'single': [
        "Primary organ involvement: {organ}.",
        "Main organ dysfunction in {organ} system.",
        "Dominant organ burden: {organ}.",
    ],
    'double': [
        "Multi-organ involvement: {org1} and {org2}.",
        "Primary dysfunction in {org1} with secondary {org2} concern.",
        "Dual organ burden affecting {org1} and {org2}.",
    ],
    'none': [
        "No specific dominant organ dysfunction identified.",
        "Organ function largely preserved at this time.",
    ],
}

VENT_TEMPLATES = {
    'vent': [
        "Patient is on mechanical ventilation.",
        "Currently mechanically ventilated.",
    ],
    'no_vent': [
        "Patient is not mechanically ventilated.",
        "Spontaneous breathing maintained.",
    ],
}

# History: vital sign trend templates
HR_HISTORY = {
    'tachycardic': [
        "Heart rate became tachycardic.",
        "Tachycardia developed.",
        "Heart rate is elevated.",
    ],
    'bradycardic': [
        "Heart rate dropped to bradycardic range.",
        "Bradycardia observed.",
        "Heart rate is abnormally low.",
    ],
    'normal': [
        "Heart rate remains normal.",
        "Heart rate is stable.",
    ],
}

BP_HISTORY = {
    'hypotensive': [
        "Blood pressure dropped to hypotensive range.",
        "Hypotension developed.",
        "Blood pressure is critically low.",
    ],
    'hypertensive': [
        "Blood pressure is elevated.",
        "Hypertension observed.",
    ],
    'stable': [
        "Blood pressure is stable.",
        "Blood pressure remains in normal range.",
    ],
}

TEMP_HISTORY = {
    'febrile': [
        "Temperature rose to febrile range.",
        "Fever developed.",
    ],
    'hypothermic': [
        "Temperature dropped to hypothermic range.",
        "Hypothermia observed.",
    ],
    'normal': [
        "Temperature is normal.",
        "Temperature remains stable.",
    ],
}

RR_HISTORY = {
    'tachypneic': [
        "Respiratory rate increased.",
        "Tachypnea observed.",
    ],
    'bradypneic': [
        "Respiratory rate is low.",
    ],
    'normal': [
        "Respiratory rate is normal.",
    ],
}

# History: lab trend templates
LACTATE_HISTORY = {
    'rising': [
        "Lactate is rising, suggesting worsening perfusion.",
        "Lactate trend is upward.",
        "Lactate increasing, tissue perfusion deteriorating.",
    ],
    'falling': [
        "Lactate is falling, perfusion improving.",
        "Lactate trend is downward.",
        "Lactate decreasing.",
    ],
    'stable': [
        "Lactate is stable.",
        "Lactate unchanged.",
    ],
}

CREATININE_HISTORY = {
    'elevated': [
        "Renal function is worsening.",
        "Creatinine rising, renal injury progressing.",
    ],
    'mildly elevated': [
        "Renal function mildly abnormal.",
        "Creatinine slightly elevated.",
    ],
    'normal': [
        "Renal function is stable.",
        "Creatinine is normal.",
    ],
}

WBC_HISTORY = {
    'elevated': [
        "White blood cell count is elevated.",
        "Leukocytosis present.",
    ],
    'low': [
        "White blood cell count is low.",
        "Leukopenia detected.",
    ],
    'normal': [
        "White blood cell count is normal.",
    ],
    'rising': [
        "White blood cell count is rising.",
        "Leukocyte trend increasing.",
    ],
    'falling': [
        "White blood cell count is falling.",
        "Leukocyte trend decreasing.",
    ],
    'stable': [
        "White blood cell count is stable.",
    ],
}

SI_HISTORY = {
    'elevated': [
        "Shock index is elevated, suggesting circulatory compromise.",
        "Shock index high, perfusion concern.",
    ],
    'normal': [
        "Shock index is normal.",
    ],
}

PFR_HISTORY = {
    'impaired': [
        "Oxygenation is impaired.",
        "Gas exchange is deteriorating.",
    ],
    'adequate': [
        "Oxygenation is adequate.",
    ],
}

FLUID_HISTORY = {
    'positive': [
        "Fluid balance is positive, fluid accumulating.",
        "Net fluid gain observed.",
    ],
    'negative': [
        "Fluid balance is negative, net fluid loss.",
        "Net fluid output exceeds input.",
    ],
    'neutral': [
        "Fluid balance is neutral.",
    ],
}

# Foresight: abnormal indicator templates
LACTATE_FORESIGHT = {
    'critically high': [
        "Lactate is critically high, severe tissue hypoperfusion.",
        "Lactate dangerously elevated, septic shock likely.",
    ],
    'elevated': [
        "Lactate is elevated, monitoring perfusion closely.",
        "Lactate above normal, perfusion concern.",
    ],
    'normal': [
        "Lactate is within normal range.",
    ],
}

PLT_FORESIGHT = {
    'critically low': [
        "Platelets are critically low, bleeding risk high.",
        "Severe thrombocytopenia, coagulopathy concern.",
    ],
    'low': [
        "Platelets are low, monitor for further decline.",
    ],
    'normal': [
        "Platelets are normal.",
    ],
}

BILI_FORESIGHT = {
    'elevated': [
        "Bilirubin is elevated, liver dysfunction.",
    ],
    'normal': [
        "Bilirubin is normal.",
    ],
}

INR_FORESIGHT = {
    'elevated': [
        "INR is elevated, coagulation impaired.",
        "Coagulopathy present.",
    ],
    'normal': [
        "INR is normal, coagulation adequate.",
    ],
}

GCS_FORESIGHT = {
    'severely impaired': [
        "GCS severely impaired, urgent neurological concern.",
        "Consciousness critically reduced, neuroprotection priority.",
    ],
    'impaired': [
        "GCS is impaired, neurological concern.",
        "Consciousness level declining.",
    ],
    'normal': [
        "GCS is normal, consciousness intact.",
    ],
}

CREATININE_FORESIGHT = {
    'elevated': [
        "Creatinine is elevated, acute kidney injury.",
    ],
    'mildly elevated': [
        "Creatinine mildly elevated, monitor renal function.",
    ],
    'normal': [
        "Creatinine is normal, renal function preserved.",
    ],
}

WBC_FORESIGHT = {
    'elevated': [
        "WBC is elevated, inflammatory response active.",
    ],
    'low': [
        "WBC is low, immunosuppression concern.",
    ],
    'normal': [
        "WBC is normal.",
    ],
}

RISK_TEMPLATES = {
    'critical': [
        "Overall risk is critical, multiple organ failures.",
    ],
    'high': [
        "Overall risk is high, significant organ dysfunction.",
    ],
    'moderate': [
        "Overall risk is moderate, some organ concerns.",
    ],
    'low': [
        "Overall risk is low, minimal organ dysfunction.",
    ],
}

FOCUS_TEMPLATES = {
    'single': [
        "Focus on protecting {organ} function.",
        "Priority: {organ} system support.",
    ],
    'double': [
        "Focus on protecting {org1} and {org2} function.",
        "Priority: {org1} and {org2} support.",
    ],
    'none': [
        "No specific organ protection priority.",
    ],
}

FIRST_HISTORY = [
    "Initial assessment: {urgency} severity sepsis. "
    "Primary dysfunction in {burden}. {vent_status}. "
    "Starting treatment with target SAPS-II improvement.",
]


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=np.float32)


def _action_description(action_id):
    return ACTION_DESCRIPTIONS.get(int(round(action_id)), f"treatment cluster {int(round(action_id))}")


def _severity_key(saps2):
    if saps2 >= 85:
        return 'high'
    elif saps2 >= 75:
        return 'moderate'
    else:
        return 'low'


def _urgency_key(saps2):
    if saps2 >= 90:
        return 'critical'
    elif saps2 >= 85:
        return 'high'
    elif saps2 >= 75:
        return 'moderate'
    else:
        return 'low'


def _burden_text(acuities_row):
    """Identify dominant organ burden from SAPS2 component subscores."""
    scores = acuities_row[3:10]
    active = [(ORGAN_NAMES[i], float(scores[i])) for i in range(len(scores)) if scores[i] > 0]
    active.sort(key=lambda x: x[1], reverse=True)
    if not active:
        return 'none', '', ''
    if len(active) == 1:
        return 'single', active[0][0], ''
    return 'double', active[0][0], active[1][0]


def _vent_text(state):
    # state[41] is mechvent: >0 means ventilated (in normalized space)
    is_vent = state[41] > 0 if len(state) > 41 else False
    key = 'vent' if is_vent else 'no_vent'
    return random.choice(VENT_TEMPLATES[key])


def _build_history_text(prev_state, curr_state, action_id):
    """Build qualitative history text: action + vital/lab trends."""
    parts = []

    # Action description
    action_desc = _action_description(action_id)
    parts.append(f"After {action_desc}:")

    # Vital sign trends (compare prev vs curr)
    if len(prev_state) >= 2 and len(curr_state) >= 2:
        hr_prev, hr_curr = prev_state[1], curr_state[1]
        sbp_prev, sbp_curr = prev_state[2], curr_state[2]
        rr_prev, rr_curr = prev_state[5], curr_state[5]
        temp_prev, temp_curr = prev_state[6], curr_state[6]

        hr_key = _classify_hr(hr_curr)
        parts.append(random.choice(HR_HISTORY[hr_key]))

        bp_key = _classify_bp(sbp_curr)
        parts.append(random.choice(BP_HISTORY[bp_key]))

        if abs(rr_curr - rr_prev) > 2:
            rr_key = _classify_rr(rr_curr)
            parts.append(random.choice(RR_HISTORY[rr_key]))

        if abs(temp_curr - temp_prev) > 0.5:
            temp_key = _classify_temp(temp_curr)
            parts.append(random.choice(TEMP_HISTORY[temp_key]))

    # Lab trends (compare prev vs curr)
    if len(prev_state) >= 25 and len(curr_state) >= 25:
        lac_delta = curr_state[24] - prev_state[24]
        lac_key = _classify_trend(lac_delta)
        parts.append(random.choice(LACTATE_HISTORY[lac_key]))

        cr_delta = curr_state[31] - prev_state[31]
        cr_key = _classify_creatinine(cr_delta)
        if cr_key != 'normal':
            parts.append(random.choice(CREATININE_HISTORY[cr_key]))

        wbc_delta = curr_state[15] - prev_state[15]
        wbc_key = _classify_trend(wbc_delta, threshold=1.0)
        if wbc_key != 'normal':
            parts.append(random.choice(WBC_HISTORY[wbc_key]))

    # Organ function trends
    if len(prev_state) >= 28 and len(curr_state) >= 28:
        si_curr = curr_state[26]
        si_key = _classify_shock_index(si_curr)
        if si_key == 'elevated':
            parts.append(random.choice(SI_HISTORY[si_key]))

        pfr_curr = curr_state[27]
        pfr_key = _classify_pfr(pfr_curr)
        if pfr_key == 'impaired':
            parts.append(random.choice(PFR_HISTORY[pfr_key]))

    if len(prev_state) >= 39 and len(curr_state) >= 39:
        bal_delta = curr_state[38] - prev_state[38] if len(curr_state) > 38 else 0
        fluid_key = _classify_fluid(bal_delta)
        if fluid_key != 'neutral':
            parts.append(random.choice(FLUID_HISTORY[fluid_key]))

    return " ".join(parts)


def _build_foresight_text(curr_state, acuities_row):
    """Build qualitative foresight text: risk assessment + organ protection."""
    concerns = []

    # Count abnormal indicators
    n_abnormal = 0

    # Lactate
    if len(curr_state) >= 25:
        lactate = curr_state[24]
        lac_key = _classify_lactate_level(lactate)
        concerns.append(random.choice(LACTATE_FORESIGHT[lac_key]))
        if lac_key != 'normal':
            n_abnormal += 1

    # Platelets
    if len(curr_state) >= 16:
        platelets = curr_state[16]
        plt_key = _classify_platelets(platelets)
        concerns.append(random.choice(PLT_FORESIGHT[plt_key]))
        if plt_key != 'normal':
            n_abnormal += 1

    # Bilirubin
    if len(curr_state) >= 35:
        bili = curr_state[34]
        bili_key = _classify_bilirubin(bili)
        concerns.append(random.choice(BILI_FORESIGHT[bili_key]))
        if bili_key != 'normal':
            n_abnormal += 1

    # INR
    if len(curr_state) >= 36:
        inr = curr_state[35]
        inr_key = _classify_inr(inr)
        concerns.append(random.choice(INR_FORESIGHT[inr_key]))
        if inr_key != 'normal':
            n_abnormal += 1

    # GCS
    gcs = curr_state[0]
    gcs_key = _classify_gcs(gcs)
    concerns.append(random.choice(GCS_FORESIGHT[gcs_key]))
    if gcs_key != 'normal':
        n_abnormal += 1

    # Creatinine
    if len(curr_state) >= 32:
        creatinine = curr_state[31]
        cr_key = _classify_creatinine(creatinine)
        if cr_key != 'normal':
            concerns.append(random.choice(CREATININE_FORESIGHT[cr_key]))
            n_abnormal += 1

    # WBC
    if len(curr_state) >= 16:
        wbc = curr_state[15]
        wbc_key = _classify_wbc(wbc)
        if wbc_key != 'normal':
            concerns.append(random.choice(WBC_FORESIGHT[wbc_key]))
            n_abnormal += 1

    # Risk level
    risk_key = _classify_risk(n_abnormal)
    concerns.append(random.choice(RISK_TEMPLATES[risk_key]))

    # Focus / protection priority
    burden_key, org1, org2 = _burden_text(acuities_row)
    if burden_key == 'none':
        concerns.append(random.choice(FOCUS_TEMPLATES['none']))
    else:
        concerns.append(random.choice(FOCUS_TEMPLATES[burden_key]).format(
            organ=org1, org1=org1, org2=org2
        ))

    return " ".join(concerns)


def build_prompt_sequences_for_trajectory(
    states: Sequence,
    acuities: Sequence,
    rtgs: Sequence,
    actions: Optional[Sequence] = None,
    rewards: Optional[Sequence] = None,
    max_timestep: int = 20,
) -> Dict[str, List[str]]:
    """
    Build clinically meaningful qualitative prompts for each timestep.

    Interface is IDENTICAL to v2 for drop-in compatibility.

    Args:
        states: (T, 45) normalized state features
        acuities: (T, 10) real SAPS2 scores [age, hr, saps2_total, comp1...comp7]
        rtgs: (T,) continuous return-to-go
        actions: (T-1,) discrete action IDs
        rewards: (T,) reward values (unused in v3, kept for interface compat)
        max_timestep: int
    """
    states_np = _to_numpy(states)
    acuities_np = _to_numpy(acuities)
    rtgs_np = _to_numpy(rtgs).reshape(-1)
    actions_np = None if actions is None else _to_numpy(actions).reshape(-1)

    T = states_np.shape[0]
    task_prompts: List[str] = []
    hindsight_prompts: List[str] = []
    foresight_prompts: List[str] = []

    for t in range(T):
        curr_saps2 = float(acuities_np[t, 2])
        rtg = float(rtgs_np[t])
        severity = _severity_key(curr_saps2)

        # --- Task Prompt ---
        urgency = _urgency_key(curr_saps2)
        task_text = random.choice(TASK_TEMPLATES[urgency])

        burden_key, org1, org2 = _burden_text(acuities_np[t])
        if burden_key == 'single':
            burden_text = random.choice(BURDEN_TEMPLATES['single']).format(organ=org1)
        elif burden_key == 'double':
            burden_text = random.choice(BURDEN_TEMPLATES['double']).format(
                org1=org1, org2=org2)
        else:
            burden_text = random.choice(BURDEN_TEMPLATES['none'])

        vent_text = _vent_text(states_np[t])

        task_prompts.append(
            f"{task_text} Severity: {severity}. {burden_text} "
            f"{vent_text} Target: cumulative SAPS-II improvement of {rtg:.1f} points."
        )

        # --- Hindsight Prompt ---
        if t > 0 and actions_np is not None:
            prev_state = states_np[t - 1]
            curr_state = states_np[t]
            action_id = actions_np[t - 1]
            hindsight_prompts.append(_build_history_text(prev_state, curr_state, action_id))
        else:
            # First timestep: use patient profile
            saps2_at_0 = float(acuities_np[0, 2])
            urgency_0 = _urgency_key(saps2_at_0)
            burden_key_0, org1_0, org2_0 = _burden_text(acuities_np[0])
            vent_0 = _vent_text(states_np[0])

            burden_str = ""
            if burden_key_0 == 'single':
                burden_str = f"Primary dysfunction in {org1_0}."
            elif burden_key_0 == 'double':
                burden_str = f"Multi-organ involvement: {org1_0} and {org2_0}."

            hindsight_prompts.append(
                f"Initial assessment: {urgency_0} severity sepsis. "
                f"{burden_str} {vent_0} Starting treatment."
            )

        # --- Foresight Prompt ---
        foresight_prompts.append(_build_foresight_text(states_np[t], acuities_np[t]))

    return {
        "task_prompts": task_prompts,
        "hindsight_prompts": hindsight_prompts,
        "foresight_prompts": foresight_prompts,
    }
