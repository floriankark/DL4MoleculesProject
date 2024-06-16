# DL4MoleculesProject
Learning from ensemble filtered data based on aggregated confidence score of multiple encoder models.

We test if [ChemBERTa](https://arxiv.org/abs/2010.09885) pretrained on MLM is able to improve when finetuned on correct prediction but low confidence samples of unseen data.
1. Collect confidence scores for all samples
2. Examine confidence distribution and set threshold for low confidence
3. Filter for correct label prediction and low confidence
4. Finetune on collected data

Next we repeat this for mid and high confidence data to have comparisons. Our baseline will be the same model finetuned on all data.

Additional (still in refinement):
- Use multiple models, predict labels and only finetune on samples if at least one or majority of models has high confidence. -> filter out outliers that hurt generalization
- Generate data with ChemGPT and use an ensemble of models to predict label. Keep high confidence samples. Finetune on these generated samples.
