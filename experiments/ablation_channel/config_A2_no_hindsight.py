# A2: no_hindsight — mask hindsight embedding, keep task + foresight
# Policy: SeMDT 6-token, hindsight slot zeroed
# WorldModel: semantic concat [task(896) | zeros(896) | foresight(896)]

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'main_model', 'scheme3_cspdt_v2'))
from config_no_sem_sigma2 import *

VARIANT = 'A2_no_hindsight'
VARIANT_TAG = 'A2_no_hindsight'
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints', VARIANT)
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
