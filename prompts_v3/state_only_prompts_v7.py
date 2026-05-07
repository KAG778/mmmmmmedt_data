"""
v7 State-Only Qualitative Prompt Builder for SeMDT Rollout.

Designed for dynamic semantic embedding generation during rollout:
  - Step 0: use v3 prompts (with acuities, richer context)
  - Step 1+: use v7 prompts (only 45-dim state, no acuities needed)

This enables online embedding generation from World Model predicted states.
"""
import random
import numpy as np
from typing import Dict, List, Optional, Sequence

# Reuse action descriptions from v3
from prompts_v3.saps2_qualitative_prompts import ACTION_DESCRIPTIONS

# --- Classification helpers (all based on normalized state features) ---

def _classify_hr(hr):
    if hr > 1.5: return 'tachycardic'
    elif hr < -1.5: return 'bradycardic'
    return 'normal'

def _classify_bp(sbp):
    if sbp < -1.5: return 'hypotensive'
    elif sbp > 1.5: return 'hypertensive'
    return 'stable'

def _classify_rr(rr):
    if rr > 1.5: return 'tachypneic'
    elif rr < -1.5: return 'bradypneic'
    return 'normal'

def _classify_temp(temp):
    if temp > 1.5: return 'febrile'
    elif temp < -1.5: return 'hypothermic'
    return 'normal'

def _classify_gcs(gcs):
    if gcs < -2.0: return 'severely impaired'
    elif gcs < -0.5: return 'impaired'
    return 'normal'

def _classify_lactate_level(lactate):
    if lactate > 2.0: return 'critically high'
    elif lactate > 0.5: return 'elevated'
    return 'normal'

def _classify_platelets(plt):
    if plt < -2.0: return 'critically low'
    elif plt < -0.5: return 'low'
    return 'normal'

def _classify_bilirubin(bili):
    if bili > 1.5: return 'elevated'
    return 'normal'

def _classify_inr(inr):
    if inr > 1.5: return 'elevated'
    return 'normal'

def _classify_creatinine(cr):
    if cr > 2.0: return 'elevated'
    elif cr > 0.5: return 'mildly elevated'
    return 'normal'

def _classify_wbc(wbc):
    if wbc > 1.5: return 'elevated'
    elif wbc < -1.5: return 'low'
    return 'normal'

def _classify_shock_index(si):
    if si > 1.0: return 'elevated'
    return 'normal'

def _classify_pfr(pfr):
    if pfr < -1.0: return 'impaired'
    return 'adequate'

def _classify_fluid(balance):
    if balance > 1.0: return 'positive'
    elif balance < -1.0: return 'negative'
    return 'neutral'

def _classify_trend(delta, threshold=0.3):
    if delta > threshold: return 'rising'
    elif delta < -threshold: return 'falling'
    return 'stable'

def _vent_text(state):
    is_vent = state[41] > 0 if len(state) > 41 else False
    return 'mechanically ventilated' if is_vent else 'spontaneous breathing'

def _assess_severity(state):
    """Estimate severity from state features alone (no SAPS2 score needed)."""
    n_abnormal = 0

    if len(state) > 24 and state[24] > 0.5: n_abnormal += 1  # Lactate
    if len(state) > 16 and state[16] < -0.5: n_abnormal += 1  # Platelets
    if len(state) > 35 and state[35] > 1.0: n_abnormal += 1   # INR
    if state[0] < -0.5: n_abnormal += 1                        # GCS
    if len(state) > 31 and state[31] > 0.5: n_abnormal += 1   # Creatinine
    if len(state) > 26 and state[26] > 1.0: n_abnormal += 1   # Shock Index
    if len(state) > 27 and state[27] < -1.0: n_abnormal += 1  # P/F ratio

    if n_abnormal >= 4: return 'critical'
    elif n_abnormal >= 3: return 'high'
    elif n_abnormal >= 1: return 'moderate'
    return 'low'

def _assess_organ_burden(state):
    """Estimate organ burden from state features."""
    concerns = []

    # Cardiovascular: shock index + BP
    if len(state) > 26 and state[26] > 1.0: concerns.append('cardiovascular')
    if state[2] < -1.0: concerns.append('cardiovascular')

    # Respiratory: P/F ratio + vent + RR
    if len(state) > 27 and state[27] < -1.0: concerns.append('respiratory')
    if len(state) > 41 and state[41] > 0: concerns.append('respiratory')

    # Renal: creatinine
    if len(state) > 31 and state[31] > 0.5: concerns.append('renal')

    # Hepatic: bilirubin + INR
    if len(state) > 34 and state[34] > 1.0: concerns.append('hepatic')
    if len(state) > 35 and state[35] > 1.0: concerns.append('hepatic')

    # Hematologic: platelets + WBC
    if len(state) > 16 and state[16] < -0.5: concerns.append('hematologic')
    if len(state) > 15 and (state[15] > 1.5 or state[15] < -1.5): concerns.append('hematologic')

    # Neurological: GCS
    if state[0] < -0.5: concerns.append('neurological')

    # Metabolic: lactate
    if len(state) > 24 and state[24] > 0.5: concerns.append('metabolic')

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in concerns:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique

# --- Template variants ---

TASK_V7_TEMPLATES = {
    'critical': [
        "CRITICAL patient condition with multiple system involvement.",
        "Life-threatening state, multiple organ concerns detected.",
        "Severe clinical deterioration, emergency response needed.",
    ],
    'high': [
        "Severe patient condition with significant organ involvement.",
        "High-acuity state requiring close monitoring.",
        "Serious clinical presentation with organ dysfunction.",
    ],
    'moderate': [
        "Moderate illness severity, some organ concerns present.",
        "Patient condition moderate, responding to management.",
        "Moderate risk state with ongoing treatment needs.",
    ],
    'low': [
        "Mild presentation, improving clinical trajectory.",
        "Low severity state, stable on current management.",
        "Minimal organ dysfunction, favorable trend.",
    ],
}

ORGAN_V7_TEMPLATES = {
    'none': [
        "No specific dominant organ dysfunction identified.",
        "Organ function largely preserved at this time.",
    ],
    'single': [
        "Primary concern: {org}.",
        "Dominant issue in {org} system.",
        "Main dysfunction: {org}.",
    ],
    'multi': [
        "Multi-system involvement: {orgs}.",
        "Several organs affected: {orgs}.",
        "Combined dysfunction in {orgs}.",
    ],
}

VENT_V7_TEMPLATES = {
    'vent': [
        "Patient is on mechanical ventilation.",
        "Currently mechanically ventilated.",
    ],
    'no_vent': [
        "Patient is not mechanically ventilated.",
        "Spontaneous breathing maintained.",
    ],
}

HINDSIGHT_V7_TEMPLATES = {
    'hr': {
        'tachycardic': ["Heart rate is elevated.", "Tachycardia present."],
        'bradycardic': ["Heart rate is low.", "Bradycardia observed."],
        'normal': ["Heart rate is stable.", "Heart rate remains normal."],
    },
    'bp': {
        'hypotensive': ["Blood pressure is low.", "Hypotension present."],
        'hypertensive': ["Blood pressure is elevated.", "Hypertension observed."],
        'stable': ["Blood pressure is stable.", "Blood pressure remains normal."],
    },
    'rr': {
        'tachypneic': ["Respiratory rate increased.", "Tachypnea observed."],
        'bradypneic': ["Respiratory rate is low."],
        'normal': ["Respiratory rate is normal."],
    },
    'temp': {
        'febrile': ["Fever present.", "Temperature elevated."],
        'hypothermic': ["Hypothermia observed.", "Temperature is low."],
        'normal': ["Temperature is normal.", "Temperature remains stable."],
    },
    'lactate': {
        'rising': ["Lactate is rising, perfusion may be worsening."],
        'falling': ["Lactate is falling, perfusion improving."],
        'stable': ["Lactate is stable."],
    },
    'creatinine': {
        'elevated': ["Renal function is worsening.", "Creatinine elevated."],
        'mildly elevated': ["Renal function mildly abnormal."],
        'normal': ["Renal function is stable."],
    },
    'wbc': {
        'rising': ["White blood cell count is rising."],
        'falling': ["White blood cell count is falling."],
        'stable': ["White blood cell count is stable."],
    },
    'shock_index': {
        'elevated': ["Shock index elevated, circulatory concern."],
        'normal': ["Shock index is normal."],
    },
    'pfr': {
        'impaired': ["Oxygenation is impaired.", "Gas exchange deteriorating."],
        'adequate': ["Oxygenation is adequate."],
    },
    'fluid': {
        'positive': ["Fluid balance positive, fluid accumulating."],
        'negative': ["Fluid balance negative, net fluid loss."],
        'neutral': ["Fluid balance is neutral."],
    },
}

FORESIGHT_V7_TEMPLATES = {
    'lactate': {
        'critically high': ["Lactate critically high, severe hypoperfusion."],
        'elevated': ["Lactate elevated, monitor perfusion."],
        'normal': ["Lactate is normal."],
    },
    'platelets': {
        'critically low': ["Platelets critically low, bleeding risk."],
        'low': ["Platelets are low, monitor closely."],
        'normal': ["Platelets are normal."],
    },
    'bilirubin': {
        'elevated': ["Bilirubin elevated, liver dysfunction."],
        'normal': ["Bilirubin is normal."],
    },
    'inr': {
        'elevated': ["INR elevated, coagulation impaired."],
        'normal': ["INR is normal."],
    },
    'gcs': {
        'severely impaired': ["Consciousness severely reduced, neuroprotection priority."],
        'impaired': ["Consciousness level declining."],
        'normal': ["Consciousness intact."],
    },
    'creatinine': {
        'elevated': ["Acute kidney injury present."],
        'mildly elevated': ["Renal function mildly abnormal, monitor."],
        'normal': ["Renal function preserved."],
    },
    'wbc': {
        'elevated': ["Inflammatory response active."],
        'low': ["Immunosuppression concern."],
        'normal': ["WBC is normal."],
    },
    'risk': {
        'critical': ["Overall risk is critical, multiple system failures."],
        'high': ["Overall risk is high, significant dysfunction."],
        'moderate': ["Overall risk is moderate, some concerns."],
        'low': ["Overall risk is low, minimal dysfunction."],
    },
    'focus': {
        'single': ["Focus on protecting {organ} function."],
        'multi': ["Priority: {organs} support."],
        'none': ["No specific organ protection priority."],
    },
}


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray): return x
    return np.asarray(x, dtype=np.float32)


def _action_description(action_id):
    return ACTION_DESCRIPTIONS.get(int(round(action_id)), f"treatment cluster {int(round(action_id))}")


def build_task_v7(curr_state) -> str:
    """Build task prompt from 45-dim state only."""
    parts = []

    severity = _assess_severity(curr_state)
    parts.append(random.choice(TASK_V7_TEMPLATES[severity]))

    organs = _assess_organ_burden(curr_state)
    if len(organs) == 0:
        parts.append(random.choice(ORGAN_V7_TEMPLATES['none']))
    elif len(organs) == 1:
        parts.append(random.choice(ORGAN_V7_TEMPLATES['single']).format(org=organs[0]))
    else:
        parts.append(random.choice(ORGAN_V7_TEMPLATES['multi']).format(orgs=', '.join(organs)))

    vent = curr_state[41] > 0 if len(curr_state) > 41 else False
    parts.append(random.choice(VENT_V7_TEMPLATES['vent' if vent else 'no_vent']))

    return " ".join(parts)


def build_hindsight_v7(prev_state, curr_state, action_id) -> str:
    """Build hindsight prompt from prev/curr state pair."""
    parts = []

    parts.append(f"After {_action_description(action_id)}:")

    # Vital signs
    hr_key = _classify_hr(curr_state[1])
    parts.append(random.choice(HINDSIGHT_V7_TEMPLATES['hr'][hr_key]))

    bp_key = _classify_bp(curr_state[2])
    parts.append(random.choice(HINDSIGHT_V7_TEMPLATES['bp'][bp_key]))

    if len(prev_state) > 5 and len(curr_state) > 5:
        rr_delta = curr_state[5] - prev_state[5]
        if abs(rr_delta) > 0.3:
            parts.append(random.choice(HINDSIGHT_V7_TEMPLATES['rr'][_classify_rr(curr_state[5])]))

    if len(prev_state) > 6 and len(curr_state) > 6:
        temp_delta = curr_state[6] - prev_state[6]
        if abs(temp_delta) > 0.3:
            parts.append(random.choice(HINDSIGHT_V7_TEMPLATES['temp'][_classify_temp(curr_state[6])]))

    # Lab trends
    if len(prev_state) > 24 and len(curr_state) > 24:
        lac_delta = curr_state[24] - prev_state[24]
        parts.append(random.choice(HINDSIGHT_V7_TEMPLATES['lactate'][_classify_trend(lac_delta)]))

    if len(prev_state) > 31 and len(curr_state) > 31:
        cr_key = _classify_creatinine(curr_state[31])
        if cr_key != 'normal':
            parts.append(random.choice(HINDSIGHT_V7_TEMPLATES['creatinine'][cr_key]))

    if len(prev_state) > 15 and len(curr_state) > 15:
        wbc_delta = curr_state[15] - prev_state[15]
        wbc_trend = _classify_trend(wbc_delta, threshold=0.5)
        if wbc_trend != 'stable':
            parts.append(random.choice(HINDSIGHT_V7_TEMPLATES['wbc'][wbc_trend]))

    # Organ function
    if len(curr_state) > 26:
        si_key = _classify_shock_index(curr_state[26])
        if si_key == 'elevated':
            parts.append(random.choice(HINDSIGHT_V7_TEMPLATES['shock_index'][si_key]))

    if len(curr_state) > 27:
        pfr_key = _classify_pfr(curr_state[27])
        if pfr_key == 'impaired':
            parts.append(random.choice(HINDSIGHT_V7_TEMPLATES['pfr'][pfr_key]))

    if len(prev_state) > 38 and len(curr_state) > 38:
        bal_delta = curr_state[38] - prev_state[38]
        fluid_key = _classify_fluid(bal_delta)
        if fluid_key != 'neutral':
            parts.append(random.choice(HINDSIGHT_V7_TEMPLATES['fluid'][fluid_key]))

    return " ".join(parts)


def build_hindsight_v7_first(curr_state) -> str:
    """First timestep hindsight (no prev state)."""
    severity = _assess_severity(curr_state)
    organs = _assess_organ_burden(curr_state)
    vent = _vent_text(curr_state)

    burden_str = "No specific organ dysfunction." if not organs else f"Concerns: {', '.join(organs)}."

    return f"Initial assessment: {severity} severity. {burden_str} {vent} Starting treatment."


def build_foresight_v7(curr_state) -> str:
    """Build foresight prompt from 45-dim state only."""
    concerns = []
    n_abnormal = 0

    # Lactate
    if len(curr_state) > 24:
        key = _classify_lactate_level(curr_state[24])
        concerns.append(random.choice(FORESIGHT_V7_TEMPLATES['lactate'][key]))
        if key != 'normal': n_abnormal += 1

    # Platelets
    if len(curr_state) > 16:
        key = _classify_platelets(curr_state[16])
        concerns.append(random.choice(FORESIGHT_V7_TEMPLATES['platelets'][key]))
        if key != 'normal': n_abnormal += 1

    # Bilirubin
    if len(curr_state) > 34:
        key = _classify_bilirubin(curr_state[34])
        concerns.append(random.choice(FORESIGHT_V7_TEMPLATES['bilirubin'][key]))
        if key != 'normal': n_abnormal += 1

    # INR
    if len(curr_state) > 35:
        key = _classify_inr(curr_state[35])
        concerns.append(random.choice(FORESIGHT_V7_TEMPLATES['inr'][key]))
        if key != 'normal': n_abnormal += 1

    # GCS
    key = _classify_gcs(curr_state[0])
    concerns.append(random.choice(FORESIGHT_V7_TEMPLATES['gcs'][key]))
    if key != 'normal': n_abnormal += 1

    # Creatinine
    if len(curr_state) > 31:
        key = _classify_creatinine(curr_state[31])
        if key != 'normal':
            concerns.append(random.choice(FORESIGHT_V7_TEMPLATES['creatinine'][key]))
            n_abnormal += 1

    # WBC
    if len(curr_state) > 15:
        key = _classify_wbc(curr_state[15])
        if key != 'normal':
            concerns.append(random.choice(FORESIGHT_V7_TEMPLATES['wbc'][key]))
            n_abnormal += 1

    # Risk level
    if n_abnormal >= 4: risk = 'critical'
    elif n_abnormal >= 3: risk = 'high'
    elif n_abnormal >= 1: risk = 'moderate'
    else: risk = 'low'
    concerns.append(random.choice(FORESIGHT_V7_TEMPLATES['risk'][risk]))

    # Focus
    organs = _assess_organ_burden(curr_state)
    if len(organs) == 0:
        concerns.append(random.choice(FORESIGHT_V7_TEMPLATES['focus']['none']))
    elif len(organs) == 1:
        concerns.append(random.choice(FORESIGHT_V7_TEMPLATES['focus']['single']).format(organ=organs[0]))
    else:
        concerns.append(random.choice(FORESIGHT_V7_TEMPLATES['focus']['multi']).format(organs=' and '.join(organs[:2])))

    return " ".join(concerns)


def build_prompt_sequences_v7(
    states: Sequence,
    acuities: Sequence = None,
    rtgs: Sequence = None,
    actions: Optional[Sequence] = None,
    rewards: Optional[Sequence] = None,
    max_timestep: int = 20,
) -> Dict[str, List[str]]:
    """
    Build v7 prompt sequences for a trajectory.
    Only requires states (45-dim). acuities/rtgs are optional (used for step 0 if provided).
    """
    states_np = _to_numpy(states)
    T = states_np.shape[0]

    task_prompts = []
    hindsight_prompts = []
    foresight_prompts = []

    for t in range(T):
        curr_state = states_np[t]

        # --- Task Prompt ---
        task_prompts.append(build_task_v7(curr_state))

        # --- Hindsight Prompt ---
        if t > 0 and actions is not None:
            prev_state = states_np[t - 1]
            action_id = actions[t - 1]
            hindsight_prompts.append(build_hindsight_v7(prev_state, curr_state, action_id))
        else:
            hindsight_prompts.append(build_hindsight_v7_first(curr_state))

        # --- Foresight Prompt ---
        foresight_prompts.append(build_foresight_v7(curr_state))

    return {
        "task_prompts": task_prompts,
        "hindsight_prompts": hindsight_prompts,
        "foresight_prompts": foresight_prompts,
    }
