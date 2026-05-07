"""
Ablation config for Policy semantic components.
Imports all hyperparameters from config_no_sem_sigma2 and adds variant definitions.

ABLATE_VARIANTS: True = zero-out that component, False = keep it.
"""
from config_no_sem_sigma2 import *  # noqa: F401,F403

ABLATE_VARIANTS = {
    # Remove one
    'no_task':       {'task': True,  'hindsight': False, 'foresight': False},
    'no_hindsight':  {'task': False, 'hindsight': True,  'foresight': False},
    'no_foresight':  {'task': False, 'hindsight': False, 'foresight': True},
    # Keep one
    'only_task':     {'task': False, 'hindsight': True,  'foresight': True},
    'only_hindsight':{'task': True,  'hindsight': False, 'foresight': True},
    'only_foresight':{'task': True,  'hindsight': True,  'foresight': False},
    # Remove all
    'no_all':        {'task': True,  'hindsight': True,  'foresight': True},
}
