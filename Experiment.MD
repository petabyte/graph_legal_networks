Hypothesis
By processing the unstructured text of the HFforLegal/case-law dataset through an AI-driven extraction tool like Graphify to build a queryable knowledge graph, and subsequently extracting connected graph features (such as preferential attachment, triangle counts, and community partitions), machine learning models can more accurately predict missing legal citations (link prediction) and identify influential case law than models relying on text alone
.
Experiments
1. Data Processing and Graph Construction
Ingestion: Feed the HFforLegal/case-law dataset into Graphify. Graphify will process the documents in passes, using deterministic AST extraction alongside LLMs to extract concepts, relationships, and design rationale
.
Edge Classification: Utilize Graphify's built-in tagging to categorize the relationships between cases and legal concepts as EXTRACTED (found directly in the source text), INFERRED (a reasonable semantic inference, provided with a confidence score), or AMBIGUOUS
.
Initial Clustering: Apply Leiden community detection, which Graphify uses to find communities based on edge density rather than relying on a separate vector database or embeddings
.
2. Dataset Balancing and Temporal Splitting
Temporal Split: Because predicting future links is the primary goal, split the case-law dataset chronologically (e.g., cases published before a specific year form the training set, and cases published during or after that year form the test set)
.
Downsampling: Real-world networks are heavily unbalanced because the number of cases that do not cite each other is vastly larger than the number of actual citations
. To avoid severe class imbalance, generate negative examples (pairs of cases with no citations between them) and downsample them so your model trains on an equal number of positive and negative examples
.
3. Connected Feature Engineering Using the structured graph, engineer distinct classes of features to feed into a binary classifier:
Basic Graph Features: Calculate the number of common neighbors (shared citations) between two cases, the "preferential attachment" score (multiplying the total number of citations each case has, under the assumption that highly-cited cases are more likely to be cited again), and the total union of neighbors
.
Triangle Metrics: Compute the number of triangles a case is part of and its local clustering coefficient, as link prediction heavily relies on triangle-based metrics to capture structural density
.
Community Detection: Extract finer-grained community structures using algorithms like Label Propagation and Louvain to test the hypothesis that cases residing in the same tight-knit community or partition are significantly more likely to cite one another
.
4. Model Evaluation
Training: Train a binary classifier—such as a Random Forest, which uses an ensemble of decision trees to predict a classification via majority vote—to predict whether a link (citation) exists between two cases
.
Metrics: Evaluate the model's accuracy, precision, and recall
. Because accuracy alone can be misleading in network predictions, plot a Receiver Operating Characteristic (ROC) curve to measure the true positive rate against the false positive rate
.
Ablation Study: Compare the Area Under the Curve (AUC) across different feature sets (e.g., basic features vs. triangle features vs. community features) to isolate which graph metrics provide the most predictive power for legal inference
.