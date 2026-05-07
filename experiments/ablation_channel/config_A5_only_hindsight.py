# A5: only_hindsight — keep only hindsight embedding, mask task + foresight
# Policy: SeMDT 6-token, task + foresight slots zeroed
# WorldModel: semantic concat [zeros(896) | hindsight(896) | zeros(896)]

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'main_model', 'scheme3_cspdt_v2'))
from config_no_sem_sigma2 import *

VARIANT = 'A5_only_hindsight'
VARIANT_TAG = 'A5_only_hindsight'
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints', VARIANT)
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
