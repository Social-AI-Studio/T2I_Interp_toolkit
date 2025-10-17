*for brainstorming; replace with README later*

# Running UCE

For now, create separate env to run UCE to prevent package vers conflicts with T2I-Interp.

Use the same model (CompVis/stable-diffusion-v1-4). Results of following commands: most images, on qualitative inspection, look mixed race rather than distinctly Black, White, or Asian. This is probably why UCE paper used qualitative assessment instead of quantitative measurement of frequencies using an AI classifier, as shown in their Figure 6. May talk with paper authors on these ideas.

```
git clone https://github.com/rohitgandikota/unified-concept-editing.git
cd unified-concept-editing
mkdir models
pip install -r requirements.txt

python3 trainscripts/uce_sd_debias.py \
--model_id 'CompVis/stable-diffusion-v1-4' \
--edit_concepts 'Doctor' \
--debias_concepts 'Black; White; Asian' \
--desired_ratios 0.3333 0.3333 0.3334 \
--num_images_per_prompt 24 \
--num_inference_steps 40 \
--max_diff 0.02 \
--max_iterations 60 \
--step_size 0.2 \
--edit_scale 1.5 \
--device 'cuda:0' \
--exp_name 'debias_race_14_v2'

python3 evalscripts/generate-images-sd.py \
  --model_id 'CompVis/stable-diffusion-v1-4' \
  --uce_model_path 'uce_models/debias_race_14_v2.safetensors' \
  --prompts_path 'data/profession1_prompts.csv' \
  --save_path 'uce_results' \
  --exp_name 'debias_race_14_v2_imgs' \
  --num_images_per_prompt 10 \
  --num_inference_steps 50 \
  --device 'cuda:0'

```

---

# add prompts folder from UCE

In UCE, the eval script `generate-images-sd.py` just reads from the "prompt" column. UCE mostly contains unlearning artist style prompts, so did not upload these now. Read using the following command:
```
prompts_path = 'data/race_prompts.csv' # should take in cmd line arg

df = pd.read_csv(prompts_path)
prompts = df.prompt
 for _, row in df.iterrows():
        prompt = str(row.prompt)

```

Guide for other data cols:
- evaluation_seed:  random seed used for a prompt when generating the image. Ensures reproducibility: same seed + model + prompt should yield the same output image
- case_number: identifier index for that prompt case, which is a unique ID to reference or track which prompt generated which image (this is just metadata)

Alternatively, can just add this folder as a submodule (though beware of vers drift): https://github.com/rohitgandikota/unified-concept-editing/tree/main/data

MIST does not have code publically available yet. Compare both UCE and MIST to ensure MIST advantages are reproducible. UCE should also be compared to TIME and MEMIT to check on new intersectional bias evaluations.