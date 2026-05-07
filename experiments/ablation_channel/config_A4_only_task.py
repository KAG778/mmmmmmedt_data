# A4: only_task — keep only task embedding, mask hindsight + foresight
# Policy: SeMDT 6-token, hindsight + foresight slots zeroed
# WorldModel: semantic concat [task(896) | zeros(896) | zeros(896)]

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'main_model', 'scheme3_cspdt_v2'))
from config_no_sem_sigma2 import *

VARIANT = 'A4_only_task'
VARIANT_TAG = 'A4_only_task'
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints', VARIANT)
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
