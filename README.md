<h1 align="center">Self-Improvement for small molecule Transformer Networks</h1>
<p align="center"><i>Project Work for the course Applications of Transformer Networks in Bio- and Cheminformatics at HHU Düsseldorf</i></p>
<div align="center">
<a href="https://github.com/floriankark/DL4MoleculesProject/stargazers"><img src="https://img.shields.io/github/stars/floriankark/DL4MoleculesProject" alt="Stars Badge"/></a>
<a href="https://github.com/floriankark/DL4MoleculesProject/network/members"><img src="https://img.shields.io/github/forks/floriankark/DL4MoleculesProject" alt="Forks Badge"/></a>
<a href="https://github.com/floriankark/DL4MoleculesProject/pulls"><img src="https://img.shields.io/github/issues-pr/floriankark/DL4MoleculesProject" alt="Pull Requests Badge"/></a>
<a href="https://github.com/floriankark/DL4MoleculesProject/issues"><img src="https://img.shields.io/github/issues/floriankark/DL4MoleculesProject" alt="Issues Badge"/></a>
<a href="https://github.com/floriankark/DL4MoleculesProject/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/floriankark/DL4MoleculesProject?color=2b9348"></a>
<a href="https://github.com/floriankark/DL4MoleculesProject/blob/master/LICENSE"><img src="https://img.shields.io/github/license/floriankark/DL4MoleculesProject?color=2b9348" alt="License Badge"/></a>
</div>
<br>
<p align="center"><i>Special thanks to DeepChem and Hugging Face! Have a look at them <a href="https://huggingface.co/docs/transformers/tasks/sequence_classification">here</a> and <a href="https://deepchem.io/">here</a></i></p>
<br>

## Overview
This repository explores self-improvement in small molecule language models, specifically ChemBERTa-2, inspired by recent findings in large language models (LLMs).

## Background
A 2022 study showed that LLMs could self-improve by generating new training data using their own outputs, leading to significant prediction quality improvements. We aim to apply a similar approach to ChemBERTa-2.

## Objective
1. Fine-tune ChemBERTa-2 for a classification task.
2. Classify new data points with the trained model.
3. Select data points with likely correct but moderately confident predictions.
4. Further train the model on this new data.

## Hypothesis
Training the model on data points where it is already making likely correct predictions (but with moderate confidence) can help it become more confident and accurate, improving performance on other input data.

## Methodology
### Confidence Criteria
- **Prediction Score:** Focus on scores with moderate confidence (not too close to 0 or 1).
- **Similarity to Training Molecules:** Measure similarity using Jaccard distance between molecule fingerprints. Select data points not too similar to the training data but still accurate.

## Dataset
- **BBB Dataset:** Binary labels for 2000+ compounds on their blood-brain barrier permeability.
- **Additional Small Molecule SMILES:** 8000 small molecule drug SMILES for generating new training data.

## Implementation Steps
1. **Fine-Tuning:** Fine-tune ChemBERTa-2 on the BBB dataset.
2. **Classifying:** Use the fine-tuned model to classify new small molecules.
3. **Selecting:** Automatically select data points with moderate confidence.
4. **Further Training:** Retrain the model on the newly selected data.

## References
1. Huang, J., et al. (2022). Large language models can self-improve. *arXiv preprint arXiv:2210.11610*.
2. Ahmad, W., et al. (2022). Chemberta-2: Towards chemical foundation models. *arXiv preprint arXiv:2209.01712*.
3. Martins, I. F., et al. (2012). A Bayesian approach to in silico blood-brain barrier penetration modeling. *Journal of Chemical Information and Modeling*, 52(6), 1686-1697.

