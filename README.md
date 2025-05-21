# Mapping the Mind of Large Language Models Through Networks of Cognitive Skills and Model Modules

This repository contains the data, scripts, and experimental framework for the paper:

## Overview

LLMNeuron provides a framework for interpreting the internal structure of Large Language Models (LLMs) through the lens of cognitive science and network theory. We map datasets to abstract cognitive skills and trace how these skills activate specific weight modules within LLMs, identifying modular specializations using gradient-based pruning and network analysis.

We construct multiple bipartite and projected networks that relate:

- Datasets ⇄ Cognitive Skills  
- Datasets ⇄ Modules  
- Cognitive Skills ⇄ Modules  
- Modules ⇄ Modules

These networks allow us to analyze skill localization, module importance, and community structure within LLMs using methods such as PCA, Louvain clustering, and spectral analysis.

## Main Contributions

- **Skill Mapping**: Align 174 multiple-choice datasets with 53 cognitive skills grounded in psychological and neuroscience literature.
- **LLM-Pruner Integration**: Use gradient-based structural pruning (Taylor approximation) to evaluate module importance per dataset.
- **Bipartite & Projection Networks**: Construct dataset-skill, dataset-module, and skill-module networks to reveal hidden structures.
- **Module Community Analysis**: Apply spectral and clustering methods to uncover modular specialization and skill co-activation patterns.
- **Biological Comparison**: Evaluate fine-tuning strategies inspired by neural localization in biological systems (e.g., human vs. avian brains).

Link to the dataset:  [Google Drive](https://drive.google.com/file/d/1icPj-ivjMHqr8VZh2wk6-Y324baK1PAp/view?usp=sharing)