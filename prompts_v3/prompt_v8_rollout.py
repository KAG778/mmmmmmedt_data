"""
v8 Prompt Builder for Rollout Evaluation.

Step 0: identical to v3 (uses real acuities)
Step 1+: uses real state (from test trajectory) + Policy predicted action
  - State: real observed state at current step (from test data)
  - Action: Policy predicted action (dynamic)
  - SAPS2: from real acuities at current step
  - Organ burden: from real acuities at current step
"""
import random
import numpy as np
from typing import Dict, List, Optional, Sequence

from prompts_v3.saps2_qualitative_prompts import (
    ACTION_DESCRIPTIONS,
    TASK_TEMPLATES,
    BURDEN_TEMPLATES,
    VENT_TEMPLATES,
    RISK_TEMPLATES,
    FOCUS_TEMPLATES,
    LACTATE_FORESIGHT,
    PLT_FORESIGHT,
    BILI_FORESIGHT,
    INR_FORESIGHT,
    GCS_FORESIGHT,
    CREATININE_FORESIGHT,
    WBC_FORESIGHT,
    _severity_key,
    _urgency_key,
    _burden_text,
    _vent_text,
    _classify_lactate_level,
    _classify_platelets,
    _classify_bilirubin,
    _classify_inr,
    _classify_gcs,
    _classify_creatinine,
    _classify_wbc,
    _classify_risk,
    _to_numpy,
)


def build_task_v8(curr_state, acuities_row, rtg):
    """Build task prompt using real state and acuities."""
    curr_saps2 = float(acuities_row[2])
    urgency = _urgency_key(curr_saps2)
    severity = _severity_key(curr_saps2)
    task_text = random.choice(TASK_TEMPLATES[urgency])

    burden_key, org1, org2 = _burden_text(acuities_row)
    if burden_key == 'single':
        burden_text = random.choice(BURDEN_TEMPLATES['single']).format(organ=org1)
    elif burden_key == 'double':
        burden_text = random.choice(BURDEN_TEMPLATES['double']).format(org1=org1, org2=org2)
    else:
        burden_text = random.choice(BURDEN_TEMPLATES['none'])

    vent_text = _vent_text(curr_state)

    return (
        f"{task_text} Severity: {severity}. {burden_text} "
        f"{vent_text} Target: cumulative SAPS-II improvement of {rtg:.1f} points."
    )


def build_hindsight_v8(action_id):
    """Build hindsight prompt. Only uses Policy's predicted action."""
    action_desc = ACTION_DESCRIPTIONS.get(int(round(action_id)), f"treatment cluster {int(round(action_id))}")
    templates = [
        f"Proceeding with {action_desc}.",
        f"Administering {action_desc}.",
        f"Current intervention: {action_desc}.",
        f"Next step: {action_desc}.",
        f"Initiating {action_desc}.",
    ]
    return random.choice(templates)


def build_foresight_v8(curr_state, acuities_row):
    """Build foresight prompt using real state and acuities.
    Identical to v3's _build_foresight_text.
    """
    concerns = []
    n_abnormal = 0

    if len(curr_state) > 24:
        key = _classify_lactate_level(curr_state[24])
        concerns.append(random.choice(LACTATE_FORESIGHT[key]))
        if key != 'normal': n_abnormal += 1

    if len(curr_state) > 16:
        key = _classify_platelets(curr_state[16])
        concerns.append(random.choice(PLT_FORESIGHT[key]))
        if key != 'normal': n_abnormal += 1

    if len(curr_state) > 34:
        key = _classify_bilirubin(curr_state[34])
        concerns.append(random.choice(BILI_FORESIGHT[key]))
        if key != 'normal': n_abnormal += 1

    if len(curr_state) > 35:
        key = _classify_inr(curr_state[35])
        concerns.append(random.choice(INR_FORESIGHT[key]))
        if key != 'normal': n_abnormal += 1

    key = _classify_gcs(curr_state[0])
    concerns.append(random.choice(GCS_FORESIGHT[key]))
    if key != 'normal': n_abnormal += 1

    if len(curr_state) > 31:
        key = _classify_creatinine(curr_state[31])
        if key != 'normal':
            concerns.append(random.choice(CREATININE_FORESIGHT[key]))
            n_abnormal += 1

    if len(curr_state) > 15:
        key = _classify_wbc(curr_state[15])
        if key != 'normal':
            concerns.append(random.choice(WBC_FORESIGHT[key]))
            n_abnormal += 1

    risk_key = _classify_risk(n_abnormal)
    concerns.append(random.choice(RISK_TEMPLATES[risk_key]))

    burden_key, org1, org2 = _burden_text(acuities_row)
    if burden_key == 'none':
        concerns.append(random.choice(FOCUS_TEMPLATES['none']))
    elif burden_key == 'single':
        concerns.append(random.choice(FOCUS_TEMPLATES['single']).format(organ=org1))
    else:
        concerns.append(random.choice(FOCUS_TEMPLATES['double']).format(
            org1=org1, org2=org2))

    return " ".join(concerns)
