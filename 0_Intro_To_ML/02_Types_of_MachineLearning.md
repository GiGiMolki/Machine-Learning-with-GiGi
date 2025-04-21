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


## 3. Based on Learning Strategy

This classification is based on *how* learning is conducted — particularly how the model uses the data, stores experience, and generalizes to new inputs. The two main strategies are:

---

### 3.1 Instance-Based Learning

Instance-Based Learning algorithms **memorize training data** and generalize only when they see new examples. No explicit training phase happens — instead, computation is deferred until prediction time (lazy learners).

- **How it works**: Store all examples, compare new input with stored examples using similarity/distance metric (like Euclidean distance).
- **Algorithms**: k-Nearest Neighbors (k-NN), Case-Based Reasoning
- **Characteristics**:
  - Fast to train, slow to predict
  - Highly interpretable
  - No model abstraction — purely data-driven
- **Challenges**: Scalability, sensitivity to noise, curse of dimensionality

> 💡 *Engineer’s Lens*: Great when model transparency and simplicity are more important than performance — not commonly used in HFT where latency is critical.

<img src="/0_Intro_To_ML/Images/image17.png" alt="Instance-Based Learning - Napkin Diagram" />

<p><strong>Figure 7:</strong> This napkin diagram illustrates Instance-Based Learning, where the model memorizes the entire dataset and defers generalization to inference time. A new input is compared to stored instances using a distance metric, and the most similar instances guide the prediction. Algorithms like k-Nearest Neighbors (k-NN) are classic examples. While interpretable and simple, this approach can become inefficient for large datasets and is less suited for low-latency environments like HFT.</p>

---

### 3.2 Model-Based Learning

Model-Based Learning algorithms **build a model of the data** during training. This model is used to make predictions efficiently during inference time. These are also known as eager learners.

- **How it works**: Train a model `f(x)` using optimization techniques to minimize error on training data.
- **Algorithms**: Linear/Logistic Regression, SVM, Decision Trees, Neural Networks
- **Characteristics**:
  - Training can be costly
  - Prediction is fast
  - Supports generalization to unseen data
- **Challenges**: Overfitting, requires hyperparameter tuning, assumptions about data

> 💡 *Engineer’s Lens*: Critical in production environments like HFT where fast inference and generalization are mandatory. Most quant ML models fall in this category.


<img src="/0_Intro_To_ML/Images/image18.png" alt="Model-Based Learning - Napkin Diagram" />

<p><strong>Figure 8:</strong> This napkin diagram illustrates Model-Based Learning, where the model is trained in advance using an optimization process. The model generalizes patterns from the data and can quickly make predictions for unseen inputs. This eager learning approach is ideal for production-grade ML systems — especially in domains like HFT, where fast inference and robust generalization are critical for real-time decision-making under uncertainty.</p>

---

> 📌 **Summary Table**

| Strategy            | Description                              | Training Phase | Inference Time | Common Use Cases              |
|---------------------|------------------------------------------|----------------|----------------|-------------------------------|
| Instance-Based      | Stores data, predicts via similarity     | None (Lazy)    | Slow           | Memory-based search, fallback |
| Model-Based         | Learns abstract model from data          | Heavy (Eager)  | Fast           | Most supervised ML tasks      |


<img src="/0_Intro_To_ML/Images/image19.png" alt="Instance-Based vs. Model-Based Learning - Napkin Diagram" />

<p><strong>Figure 9:</strong> This napkin diagram compares Instance-Based and Model-Based Learning approaches. On the left, instance-based models like k-NN store all training data and make predictions by comparing new inputs to past examples — a lazy approach that requires no training phase. On the right, model-based learning involves building a predictive function during training, enabling fast and generalized inference. This contrast is crucial when considering latency, memory, and scalability — especially in real-time ML applications like high-frequency trading systems.</p>


## 4. Based on Production Mode

This classification focuses on **how the model receives and processes data over time**, especially in production environments. It's particularly critical in systems where **data streams**, **low latency**, and **continuous learning** matter — like real-time analytics, recommendation engines, or high-frequency trading systems.

---

### 4.1 Online Learning

Online Learning algorithms update the model **incrementally** as each new data point arrives. Ideal for non-stationary environments or where real-time decision-making is essential.

- **How it works**: Model is updated after every data point or mini-batch.
- **Algorithms**: Stochastic Gradient Descent, Passive-Aggressive, Online Perceptron
- **Characteristics**:
  - Fast, memory-efficient
  - Adapts to new data instantly
  - Robust to concept drift (changes in data distribution)
- **Use Cases**: High-Frequency Trading, Fraud Detection, Real-time Recommendations

> ⚙️ *Engineering Note*: Online models are essential in HFT where millisecond-level adaptation to market data is critical.

<img src="/0_Intro_To_ML/Images/image20.png" alt="Online Learning - Napkin Diagram" />

<p><strong>Figure 10:</strong> This napkin diagram represents Online Learning, where models continuously update with each new data point. It’s essential in non-stationary, low-latency environments like high-frequency trading. As data flows in real time — such as tick-by-tick market data — the model adapts immediately without retraining on the full dataset. This enables highly responsive decision-making, making it a foundational method for streaming AI systems.</p>

---

### 4.2 Offline (Batch) Learning

Offline Learning (also called Batch Learning) requires the **entire dataset beforehand** to train the model. Once trained, the model does not learn from new data until retrained.

- **How it works**: Data is processed in bulk, model is trained all at once.
- **Algorithms**: Linear Regression, SVM, Decision Trees (standard implementations)
- **Characteristics**:
  - Stable, robust training
  - Needs full data access
  - Retraining is computationally expensive
- **Use Cases**: Risk Modeling, Portfolio Optimization, Static Forecasting

> ⚙️ *Engineering Note*: Offline learning is appropriate when model stability is more important than adaptivity — such as in quarterly portfolio rebalancing or long-horizon backtests.

<img src="/0_Intro_To_ML/Images/image21.png" alt="Offline (Batch) Learning - Napkin Diagram" />

<p><strong>Figure 11:</strong> This napkin diagram illustrates Offline (Batch) Learning, where models are trained on complete historical datasets in one go. The model is fixed post-training and cannot adapt until explicitly retrained with new data. This approach is widely used in stable systems like risk analysis and portfolio optimization, where consistent behavior is preferred over adaptability. Unlike online learning, it suits environments with infrequent data updates and low tolerance for model volatility.</p>

---

 ### 🔁 Summary Table

| Mode     | Learning Style     | Memory Usage | Adaptivity | Training Frequency | Common Use Cases              |
|----------|--------------------|--------------|------------|--------------------|-------------------------------|
| Online   | Incremental (Live) | Low          | High       | Continuous         | HFT, Real-Time Bidding, IoT   |
| Offline  | Batch (Static)     | High         | Low        | Periodic           | Risk Models, Backtesting      |


<h3>🔁 Summary: Online vs Offline Learning</h3>
<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>Online Learning</th>
      <th>Offline (Batch) Learning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Learning Style</strong></td>
      <td>Incremental (one data point at a time)</td>
      <td>All at once (entire dataset at once)</td>
    </tr>
    <tr>
      <td><strong>Adaptability</strong></td>
      <td>High – adapts in real-time</td>
      <td>Low – static after training</td>
    </tr>
    <tr>
      <td><strong>Training Frequency</strong></td>
      <td>Continuous</td>
      <td>Periodic (scheduled retraining)</td>
    </tr>
    <tr>
      <td><strong>Memory Usage</strong></td>
      <td>Low – no need to store all data</td>
      <td>High – stores and reprocesses full dataset</td>
    </tr>
    <tr>
      <td><strong>Compute Cost</strong></td>
      <td>Lightweight and ongoing</td>
      <td>Heavy but done in batches</td>
    </tr>
    <tr>
      <td><strong>Latency Tolerance</strong></td>
      <td>Very low – designed for real-time</td>
      <td>Higher latency acceptable</td>
    </tr>
    <tr>
      <td><strong>Common Use Cases</strong></td>
      <td>High-Frequency Trading, Fraud Detection</td>
      <td>Portfolio Optimization, Risk Modeling</td>
    </tr>
    <tr>
      <td><strong>Examples</strong></td>
      <td>Online SGD, Passive-Aggressive, Online Perceptron</td>
      <td>SVM, Decision Trees, XGBoost</td>
    </tr>
  </tbody>
</table>


<h2>5. Based on Objective or Functionality</h2>

<h3>5.1 Predictive</h3>
<p>Predictive modeling focuses on learning a function from historical data that can make accurate predictions on unseen inputs. It is typically supervised learning, where the model maps inputs to outputs. Examples include regression models predicting stock prices, or classifiers detecting fraud.</p>

<h3>5.2 Descriptive</h3>
<p>Descriptive modeling identifies patterns or structures in data without explicitly predicting outcomes. It is primarily unsupervised and used to gain insight or group data meaningfully. Examples include clustering customer segments or discovering latent topics in documents.</p>

<h4>🔁 Predictive vs Descriptive (Comparison)</h4>
<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>Predictive</th>
      <th>Descriptive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Goal</strong></td>
      <td>Predict future or unknown outcomes</td>
      <td>Understand structure or patterns in data</td>
    </tr>
    <tr>
      <td><strong>Learning Type</strong></td>
      <td>Supervised</td>
      <td>Unsupervised</td>
    </tr>
    <tr>
      <td><strong>Examples</strong></td>
      <td>Regression, Classification</td>
      <td>Clustering, Topic Modeling</td>
    </tr>
    <tr>
      <td><strong>Real-World Use</strong></td>
      <td>Predict stock prices, detect fraud</td>
      <td>Group customers, analyze behavior</td>
    </tr>
  </tbody>
</table>

<h3>5.3 Generative</h3>
<p>Generative models learn the joint probability distribution P(X, Y) or just P(X), allowing them to generate new data instances. They model how data is generated, making them suitable for tasks like text generation, image synthesis, and data augmentation. Examples: Naive Bayes, GANs, VAEs, GPT.</p>

<h3>5.4 Discriminative</h3>
<p>Discriminative models learn the conditional probability P(Y | X) or a direct decision boundary between classes. They are optimized for classification accuracy and are often more efficient in making predictions. Examples: Logistic Regression, SVM, Decision Trees, BERT (for classification).</p>

<h4>🔁 Generative vs Discriminative (Comparison)</h4>
<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>Generative</th>
      <th>Discriminative</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Probability Modeled</strong></td>
      <td>P(X, Y) or P(X)</td>
      <td>P(Y | X)</td>
    </tr>
    <tr>
      <td><strong>Goal</strong></td>
      <td>Generate or simulate data</td>
      <td>Classify or predict labels</td>
    </tr>
    <tr>
      <td><strong>Output</strong></td>
      <td>New samples, latent representations</td>
      <td>Labels, boundaries, probabilities</td>
    </tr>
    <tr>
      <td><strong>Examples</strong></td>
      <td>Naive Bayes, GANs, GPT</td>
      <td>Logistic Regression, SVM, BERT</td>
    </tr>
  </tbody>
</table>

<h3>5.5 When to Use What (With Real-World Examples)</h3>
<ul>
  <li><strong>Use Predictive models</strong> when you need to forecast, classify, or make data-driven decisions — e.g., predicting stock prices, identifying customer churn, or real-time fraud detection in fintech.</li>
  <li><strong>Use Descriptive models</strong> when the goal is to explore or understand data without a specific label — e.g., clustering trades by behavior, or segmenting customer portfolios.</li>
  <li><strong>Use Generative models</strong> when creating new content or simulating scenarios is useful — e.g., generating synthetic market data, simulating news for sentiment analysis, or building realistic agent-based models.</li>
  <li><strong>Use Discriminative models</strong> when classification accuracy and decision boundaries are key — e.g., deciding whether to execute a trade based on features, or labeling market sentiment from news headlines.</li>
</ul>