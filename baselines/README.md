*for brainstorming; replace with README later*

Current task: add prompts folder from UCEand give instructions on how to convert to text

In UCE, the eval script `generate-images-sd.py` just reads from the "prompt" column. UCE mostly contains unlearning artist style prompts, so did not upload these now. Read usuing the following command:
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