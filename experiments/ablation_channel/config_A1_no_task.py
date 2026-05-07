# A1: no_task — mask task embedding, keep hindsight + foresight
# Policy: SeMDT 6-token, task slot zeroed
# WorldModel: semantic concat [zeros(896) | hindsight(896) | foresight(896)]

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'main_model', 'scheme3_cspdt_v2'))
from config_no_sem_sigma2 import *  # inherit all defaults

VARIANT = 'A1_no_task'
VARIANT_TAG = 'A1_no_task'
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints', VARIANT)
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
