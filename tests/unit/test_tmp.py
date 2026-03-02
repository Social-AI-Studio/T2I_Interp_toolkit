import sys
from unittest.mock import patch
import t2i_interp.scripts.run_steer as run_steer

def test_steer():
    with patch('t2i_interp.t2i.T2IModel') as MockModel, \
         patch('t2i_interp.scripts.run_steer.load_dataset') as MockLoadDS, \
         patch('t2i_interp.scripts.run_steer.collect_latents') as MockCollect, \
         patch('t2i_interp.utils.T2I.buffer.ActivationsDataloader') as MockLoader, \
         patch('t2i_interp.utils.training.Training') as MockTraining, \
         patch('t2i_interp.linear_steering.KSteer') as MockKSteer, \
         patch('t2i_interp.linear_steering.CAA') as MockCAA, \
         patch('os.makedirs'):
         
        with patch.object(sys, 'argv', ['run_steer.py', 'model_key=test', 'device=cpu', 'steer_type=caa', 'max_samples=2']):
            run_steer.main()
        
test_steer()
print("Success!")
