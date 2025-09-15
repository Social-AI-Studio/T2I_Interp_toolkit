import ray
import os
from utils.inference import InferenceSpec, Inference

ray.init(runtime_env={"working_dir": os.getcwd()}, logging_level="DEBUG")

class RayWorker:
    def run_one_inference(self,spec_dict, base_kwargs_dict):
        spec = InferenceSpec(**spec_dict)
        infer = Inference(inference_spec=spec, **base_kwargs_dict)
        return infer.predict()

    def run_multiple_inferences(self,specs, base_kwargs):    
        futures = [self.run_one_inference.remote(spec.__dict__, base_kwargs) for spec in specs]
        results = ray.get(futures)
        return results
