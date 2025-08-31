import pandas as pd
import re
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk import download as nltk_download
import os
import joblib
from tqdm import tqdm

# Dataset Preprocessing
def load_and_prepare_dataset(file_path):
    """
    Load the dataset and prepare it for processing.
    Maps labels to binary values for classification.
    """
    print(f"Loading dataset from {file_path}...")
    columns = [
        "id", "label", "statement", "subjects", "speaker", "job_title", "state_info",
        "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts",
        "mostly_true_counts", "pants_on_fire_counts", "context"
    ]
    data = pd.read_csv(file_path, sep="\t", header=None, names=columns)
    data = data[['statement', 'label']].dropna()
    label_mapping = {
        "true": 1, "mostly-true": 1,
        "half-true": 1, "barely-true": 0,
        "false": 0, "pants-fire": 0
    }
    data['label'] = data['label'].map(label_mapping)
    print(f"Dataset loaded with {len(data)} records.")
    return data['statement'], data['label']

# Text Preprocessing
def clean_text_data(text):
    """
    Preprocess text by removing URLs, email addresses, non-alphabetic characters,
    and stopwords. Normalizes case and removes excessive whitespace.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk_download('stopwords')
        stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text

# Generate Sentence Embeddings
def compute_sentence_embeddings(sentences, model, batch_size=32):
    """
    Generate embeddings for a list of sentences using a pre-trained SBERT model.
    Processes sentences in batches to optimize memory usage.
    """
    print(f"Generating embeddings for {len(sentences)} sentences...")
    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i + batch_size]
        embeddings.append(model.encode(batch))
    embeddings = np.vstack(embeddings)
    print(f"Embeddings generated with shape: {embeddings.shape}")
    return embeddings

# Objective Function for Optimization
def evaluate_classification_model(params, X_train, y_train, X_valid, y_valid):
    """
    Evaluate the performance of an ensemble model using given hyperparameters.
    Returns the negative accuracy score as the objective value.
    """
    svc_c, xgb_lr = params
    try:
        svc = SVC(C=svc_c, kernel="linear", probability=True, random_state=42)
        xgb = XGBClassifier(learning_rate=xgb_lr, eval_metric="logloss", random_state=42, tree_method="gpu_hist", use_label_encoder=False)
        ensemble = VotingClassifier(estimators=[('svm', svc), ('xgb', xgb)], voting='soft', n_jobs=-1)
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_valid)
        return -accuracy_score(y_valid, y_pred)
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return float('inf')

# Dispersive Flies Optimization
class DispersiveFliesOptimization:
    """
    An implementation of the Dispersive Flies Optimization algorithm
    for hyperparameter tuning.
    """
    def __init__(self, objective_function, num_flies, dimensions, lower_bounds, upper_bounds, max_iterations=5, disturbance_threshold=0.001):
        self.objective_function = objective_function
        self.num_flies = num_flies
        self.dimensions = dimensions
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.max_iterations = max_iterations
        self.disturbance_threshold = disturbance_threshold
        self.population = self._initialize_population()

    def _initialize_population(self):
        """
        Initialize the population of flies with random positions within bounds.
        """
        return self.lower_bounds + np.random.rand(self.num_flies, self.dimensions) * (self.upper_bounds - self.lower_bounds)

    def optimize(self):
        """
        Perform the optimization process to minimize the objective function.
        Returns the best parameters and fitness value.
        """
        print("Starting Dispersive Flies Optimization...")
        best_positions = self.population.copy()
        best_fitness = np.array([self.objective_function(p) for p in self.population])
        global_best_position = best_positions[np.argmin(best_fitness)]
        global_best_fitness = min(best_fitness)

        for iteration in range(self.max_iterations):
            for i in range(self.num_flies):
                for d in range(self.dimensions):
                    best_neighbor = best_positions[np.argmin([self.objective_function(p) for p in best_positions])]
                    self.population[i, d] += np.random.rand() * (best_neighbor[d] - self.population[i, d]) + np.random.rand() * (global_best_position[d] - self.population[i, d])
                    if np.random.rand() < self.disturbance_threshold:
                        self.population[i, d] = self.lower_bounds[d] + np.random.rand() * (self.upper_bounds[d] - self.lower_bounds[d])

            best_fitness = np.array([self.objective_function(p) for p in self.population])
            global_best_position = self.population[np.argmin(best_fitness)]
            global_best_fitness = min(best_fitness)

        print(f"Optimization completed. Best fitness: {global_best_fitness:.4f}")
        return global_best_position, global_best_fitness

# Train and Evaluate Model
def train_and_evaluate_pipeline(train_path, valid_path, test_path):
    """
    Full training and evaluation pipeline including preprocessing, embedding,
    hyperparameter optimization, and evaluation.
    """
    # Create cache directory
    cache_dir = './cache'
    os.makedirs(cache_dir, exist_ok=True)

    # Load and preprocess datasets
    X_train, y_train = load_and_prepare_dataset(train_path)
    X_valid, y_valid = load_and_prepare_dataset(valid_path)
    X_test, y_test = load_and_prepare_dataset(test_path)

    X_train = X_train.apply(clean_text_data)
    X_valid = X_valid.apply(clean_text_data)
    X_test = X_test.apply(clean_text_data)

    # Load or generate embeddings
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    X_train_embeddings = compute_sentence_embeddings(X_train.tolist(), sbert_model)
    X_valid_embeddings = compute_sentence_embeddings(X_valid.tolist(), sbert_model)
    X_test_embeddings = compute_sentence_embeddings(X_test.tolist(), sbert_model)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_embeddings, y_train)

    # Hyperparameter optimization
    def wrapped_objective(params):
        return evaluate_classification_model(params, X_train_resampled, y_train_resampled, X_valid_embeddings, y_valid)

    optimizer = DispersiveFliesOptimization(
        objective_function=wrapped_objective,
        num_flies=3,
        dimensions=2,
        lower_bounds=[0.1, 0.01],
        upper_bounds=[10, 0.3]
    )
    best_params, _ = optimizer.optimize()
    svc_c, xgb_lr = best_params

    # Train final model with optimal parameters
    svc = SVC(C=svc_c, kernel="linear", probability=True, random_state=42)
    xgb = XGBClassifier(learning_rate=xgb_lr, eval_metric="logloss", use_label_encoder=False, tree_method="gpu_hist", random_state=42)
    ensemble = VotingClassifier(estimators=[('svm', svc), ('xgb', xgb)], voting='soft', n_jobs=-1)
    ensemble.fit(X_train_resampled, y_train_resampled)

    # Evaluate final model
    y_pred = ensemble.predict(X_test_embeddings)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, ['Fake', 'Real'])

# Plot Confusion Matrix
def plot_confusion_matrix(cm, classes):
    """
    Plot a confusion matrix for model evaluation.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Main Entry Point
if __name__ == "__main__":
    train_path = 'train.tsv'
    valid_path = 'valid.tsv'
    test_path = 'test.tsv'
    train_and_evaluate_pipeline(train_path, valid_path, test_path)
