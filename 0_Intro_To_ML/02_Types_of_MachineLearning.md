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


## 2. Based on Supervision

The most fundamental classification of Machine Learning algorithms is based on the presence or absence of **supervision** — that is, whether the training data contains labeled outputs.

### 2.1 Supervised Learning

In supervised learning, the algorithm is trained on a dataset that includes input-output pairs. The goal is to learn a mapping function `f(x) = y` that can predict the output for new inputs.

- **Use Cases**: Stock price prediction, fraud detection, credit scoring
- **Common Algorithms**: Linear Regression, Random Forests, XGBoost, Support Vector Machines, Neural Networks
- **Characteristics**:
  - Requires large amounts of labeled data
  - Optimized using loss functions (e.g., MSE, Cross-Entropy)
  - Evaluated using metrics like accuracy, F1-score, AUC, etc.

> 💡 *Engineer’s Lens*: Supervised learning works well in domains with historical market data where ground truth exists (e.g., stock prices or executed trade outcomes). In HFT, it’s used for signal generation and short-term trend prediction.


<img src="/0_Intro_To_ML/Images/image12.png" alt="Supervised Learning Flow - Napkin Diagram" />

<p><strong>Figure 2:</strong> This napkin diagram illustrates the flow of supervised learning. It starts with labeled training data (input features and known outputs), which is passed into a model. The model generates predictions, which are compared against actual labels using a loss function. The model parameters are updated via optimization, forming a feedback loop. This setup is foundational for predictive tasks such as market signal forecasting in quant finance.</p>

---

### 2.2 Unsupervised Learning

Here, the dataset contains inputs but no labeled outputs. The objective is to uncover hidden patterns or structures within the data.

- **Use Cases**: Market regime detection, anomaly detection, customer segmentation
- **Common Algorithms**: K-Means, Hierarchical Clustering, DBSCAN, PCA
- **Characteristics**:
  - Works without labeled data
  - Often exploratory and hard to evaluate
  - Focuses on structure, similarity, and density

> 💡 *Engineer’s Lens*: In trading systems, unsupervised learning can cluster correlated assets or detect unusual order book behavior without needing predefined labels.

<img src="/0_Intro_To_ML/Images/image13.png" alt="Unsupervised Learning Flow - Napkin Diagram" />

<p><strong>Figure 3:</strong> This napkin diagram represents unsupervised learning where the algorithm receives unlabeled input data and attempts to discover hidden patterns. The diagram includes typical unsupervised tasks like clustering and dimensionality reduction. This approach is vital in quant finance for detecting market regimes, identifying anomalies in trading behavior, and clustering similar financial instruments without labeled outcomes.</p>

---

### 2.3 Semi-Supervised Learning

This is a hybrid approach that uses a small amount of labeled data and a large amount of unlabeled data. It’s especially useful when labels are expensive or time-consuming to obtain.

- **Use Cases**: Sentiment classification with few labeled texts, predictive modeling on sparse financial datasets
- **Common Techniques**: Label propagation, self-training, consistency regularization (e.g., FixMatch)
- **Characteristics**:
  - Balances cost-efficiency and accuracy
  - Often bootstraps from initial supervised signals
  - Assumes structure in unlabeled data can improve generalization

> 💡 *Engineer’s Lens*: In finance, this is valuable when limited annotated data (e.g., labeled news sentiment) can be expanded with large-scale unlabeled data (news headlines, Twitter feeds, etc.).

<img src="/0_Intro_To_ML/Images/image14.png" alt="Semi-Supervised Learning Flow - Napkin Diagram" />

<p><strong>Figure 4:</strong> This napkin diagram illustrates Semi-Supervised Learning, where both a small labeled dataset and a large unlabeled dataset are used for model training. The diagram emphasizes how unlabeled data can enhance learning, helping the model generalize better with less labeled data. Semi-Supervised Learning is valuable in quant finance when acquiring labeled data (e.g., for sentiment analysis or trade classification) is expensive or scarce, while unlabeled data (e.g., market data) is abundant.</p>

---

### 2.4 Reinforcement Learning

In RL, an agent learns to make decisions by interacting with an environment and receiving feedback in the form of rewards or penalties.

- **Use Cases**: Optimal trade execution, portfolio optimization, market-making strategies
- **Core Concepts**: States, actions, rewards, policy, environment
- **Common Algorithms**: Q-Learning, Deep Q-Networks (DQN), Policy Gradient, PPO
- **Characteristics**:
  - No explicit dataset — learning happens via interaction
  - Focuses on sequential decision-making
  - Often requires simulation environments or live systems

> 💡 *Engineer’s Lens*: RL is particularly suited for real-time trading agents that must learn and adapt under uncertainty, latency constraints, and adversarial conditions.

<img src="/0_Intro_To_ML/Images/image15.png" alt="Reinforcement Learning Flow - Napkin Diagram" />
<p><strong>Figure 5:</strong> This napkin diagram represents Reinforcement Learning (RL), where an agent interacts with an environment and learns to make decisions by receiving rewards or penalties. The agent aims to improve its policy over time to maximize cumulative rewards. RL is a powerful technique in Quant Finance, often used for tasks like portfolio optimization, algorithmic trading, and optimal execution in HFT, where decisions are sequential and must adapt in real-time based on market feedback.</p>

---

> 📌 **Summary Table**

| Type                | Requires Labels      | Learns From                   | Key Applications              | Challenge                               |
|---------------------|----------------------|-------------------------------|-------------------------------|-----------------------------------------|
| Supervised          | ✅ Yes               | Historical input-output pairs | Prediction, classification    | Label availability, overfitting          |
| Unsupervised        | ❌ No                | Data structure                | Clustering, anomaly detection | Evaluation, pattern validation           |
| Semi-Supervised     | 🟡 Partial           | Mixed data                    | When labeled data is scarce   | Assumes structure in unlabeled data      |
| Reinforcement       | ❌ No (uses rewards) | Interaction with environment  | Real-time decision making     | Exploration-exploitation, convergence    |


<img src="/0_Intro_To_ML/Images/image16.png" alt="Types of Machine Learning - Summary Table" />

<p><strong>Figure 6:</strong> This napkin diagram summarizes the key types of machine learning based on supervision. It includes a comparison of Supervised Learning, Unsupervised Learning, Semi-Supervised Learning, and Reinforcement Learning, with a summary table that highlights key distinctions in data requirements, training methods, and real-world applications. This table is especially useful for selecting the right approach for various tasks in Quant Finance and High-Frequency Trading, where trade-offs between data availability, latency, and real-time decision-making are critical.</p>