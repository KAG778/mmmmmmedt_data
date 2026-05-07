# A6: only_foresight — keep only foresight embedding, mask task + hindsight
# Policy: SeMDT 6-token, task + hindsight slots zeroed
# WorldModel: semantic concat [zeros(896) | zeros(896) | foresight(896)]

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'main_model', 'scheme3_cspdt_v2'))
from config_no_sem_sigma2 import *

VARIANT = 'A6_only_foresight'
VARIANT_TAG = 'A6_only_foresight'
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints', VARIANT)
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
