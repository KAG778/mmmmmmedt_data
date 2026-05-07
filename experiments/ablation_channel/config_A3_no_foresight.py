# A3: no_foresight — mask foresight embedding, keep task + hindsight
# Policy: SeMDT 6-token, foresight slot zeroed
# WorldModel: semantic concat [task(896) | hindsight(896) | zeros(896)]

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'main_model', 'scheme3_cspdt_v2'))
from config_no_sem_sigma2 import *

VARIANT = 'A3_no_foresight'
VARIANT_TAG = 'A3_no_foresight'
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints', VARIANT)
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
