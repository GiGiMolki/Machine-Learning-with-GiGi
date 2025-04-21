# Types of Machine Learning 

> Focus: Systematic classification of ML types from multiple practical dimensions, followed by engineering challenges and robust design patterns to address them.

## 1. Overview

Machine Learning is not a monolithic concept — it's a vast field with multiple learning paradigms, strategies, and deployment approaches. As engineers, especially those working in real-time systems like High-Frequency Trading (HFT), it's crucial to categorize ML not just academically, but by how it integrates into production pipelines, handles data streams, and adapts to changing environments.

We classify ML systems through different lenses:

- **Supervision**: Whether labels are available during training (Supervised, Unsupervised, Semi-supervised, Reinforcement Learning)
- **Learning Strategy**: Whether learning is instance-based or model-based
- **Training Timeline**: Whether models learn in batch (offline) or adapt continuously (online)
- **Functionality**: Predictive vs. Descriptive vs. Generative tasks
- **Deployment Context**: Latency sensitivity, data volatility, and model retrainability

In the context of **Quant Finance and HFT**, choosing the right ML paradigm affects everything from prediction latency to strategy adaptability, compute resource allocation, and explainability to regulators.

This section explores each type of ML across these dimensions, highlights engineering challenges, and proposes practical patterns to build robust ML systems at scale.

>  💡 *Engineer’s Lens*: Think of this as a blueprint to architect ML systems that don’t just work in Jupyter notebooks, but in real-world, latency-bound, mission-critical environments like algorithmic trading engines.

<img src="/0_Intro_To_ML/Images/image11.png" alt="Types of Machine Learning - Classification Overview" />
<p><strong>Figure 1:</strong> A multi-dimensional classification of Machine Learning paradigms, organized by supervision, learning strategy, timeline, functionality, and deployment context. This visual helps engineers select the right approach based on both data characteristics and real-world system constraints like latency and adaptability.</p>