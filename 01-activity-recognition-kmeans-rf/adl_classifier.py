import numpy as np
import pandas as pd
import os
import glob
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def load_data(data_path):
    """
    Load the ADL dataset from the given path.
    Returns a list of dictionaries, each containing acceleration data and activity label.
    """
    all_data = []
    
   
    if not os.path.exists(data_path):
        print(f"Error: Data path '{data_path}' does not exist!")
        return all_data
    
    # List of all activities in the dataset (folders in the HMP_dataset directory)
    activities = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    if not activities:
        print(f"Error: No activity directories found in '{data_path}'")
        print(f"Contents of directory: {os.listdir(data_path)}")
        return all_data
    
    print(f"Found {len(activities)} activities: {activities}")
    
    
    for activity in activities:
        activity_path = os.path.join(data_path, activity)
        
        
        data_files = glob.glob(os.path.join(activity_path, "*.*"))
        
        if not data_files:
            print(f"Warning: No data files found in {activity_path}")
            continue
        
        print(f"Found {len(data_files)} files for activity: {activity}")
        
        # Loop through each data file
        for data_file in data_files:
            try:
                
                # Assuming the format is one sample per line with comma or space-separated values
                accel_data = []
                
                with open(data_file, 'r') as f:
                    for line in f:
                   
                        values = line.strip().split(',')
                        
                  
                        if len(values) <= 1:
                            values = line.strip().split()
                        
                     
                        if len(values) >= 3:  # We expect at least x, y, z values
                            accel_data.append([float(v) for v in values[:3]])  
                
               
                accel_data = np.array(accel_data)
                
                
                if len(accel_data) == 0:
                    print(f"Warning: No valid data in {data_file}")
                    continue
                
                
                all_data.append({
                    'activity': activity,
                    'data': accel_data,
                    'subject': os.path.basename(data_file)
                })
                
            except Exception as e:
                print(f"Error loading {data_file}: {e}")
    
    print(f"Loaded {len(all_data)} samples")
    return all_data

# Step 2: Segment the data into fixed length samples
def segment_data(data_list, segment_length=32):
    """
    Break acceleration data into fixed-length segments.
    Returns a list of segments and their corresponding activity labels.
    """
    if not data_list:
        print("Error: No data to segment!")
        return np.array([]), np.array([])
    
    segments = []
    activities = []
    
    for item in tqdm(data_list, desc="Segmenting data"):
       
        accel_data = item['data']
        
        # Ensure the data is at least as long as the segment length
        if len(accel_data) < segment_length:
            print(f"Warning: Sample for {item['activity']} too short ({len(accel_data)} < {segment_length})")
            continue
        
        # Break into segments
        for i in range(0, len(accel_data) - segment_length, segment_length):
            segment = accel_data[i:i+segment_length]
            segments.append(segment)
            activities.append(item['activity'])
    
    print(f"Created {len(segments)} segments from {len(data_list)} samples")
    
    if not segments:
        print("Error: No segments created!")
        return np.array([]), np.array([])
    
    return np.array(segments), np.array(activities)

# Step 3: Hierarchical K-means clustering
def hierarchical_kmeans(segments, level1_k=40, level2_k=12):
    """
    Perform hierarchical k-means clustering as described in the textbook.
    Returns the trained level2 cluster model and the cluster assignments for all segments.
    """
    if segments.size == 0:
        print("Error: No segments provided for clustering!")
        return np.array([]), 0
    
    # Checking dimensions and reshape segments for clustering
    if len(segments.shape) != 3:
        print(f"Error: Expected 3D array for segments, got shape {segments.shape}")
        return np.array([]), 0
    
    n_segments, n_samples, n_dims = segments.shape
    print(f"Segments shape: {segments.shape}")
    flat_segments = segments.reshape(n_segments, -1)
    
    # Adjustisng k if required
    level1_k = min(level1_k, n_segments // 2)
    if level1_k < 2:
        print(f"Warning: Too few segments ({n_segments}) for clustering. Need at least 4.")
        return np.arange(n_segments), n_segments  # Just assign each segment to its own cluster
    
    print(f"Performing level 1 clustering with {level1_k} clusters...")
    # Level 1 clustering
    kmeans_l1 = KMeans(n_clusters=level1_k, random_state=42, n_init=10)
    cluster_labels_l1 = kmeans_l1.fit_predict(flat_segments)
    
    # Level 2 clustering
    kmeans_models_l2 = []
    cluster_assignments = np.zeros(n_segments, dtype=int)
    global_cluster_idx = 0
    
    print(f"Performing level 2 clustering with up to {level2_k} clusters per level 1 cluster...")
    for i in range(level1_k):
        # Get segments assigned to this level 1 cluster
        mask = (cluster_labels_l1 == i)
        count = np.sum(mask)
        
        if count < level2_k:
            # If too few samples, assign all to the same level 2 cluster
            print(f"  Cluster {i}: only {count} samples, skipping level 2")
            cluster_assignments[mask] = global_cluster_idx
            global_cluster_idx += 1
            continue
            
        # Cluster the segments in this level 1 cluster
        local_k = min(level2_k, count // 2)
        if local_k < 2:
            local_k = 1
        
        print(f"  Cluster {i}: {count} samples, using {local_k} level 2 clusters")
        kmeans_l2 = KMeans(n_clusters=local_k, random_state=42, n_init=10)
        cluster_labels_l2 = kmeans_l2.fit_predict(flat_segments[mask])
        
        # Assign global cluster indices
        for j in range(local_k):
            submask = mask.copy()
            submask[mask] = (cluster_labels_l2 == j)
            cluster_assignments[submask] = global_cluster_idx
            global_cluster_idx += 1
        
        kmeans_models_l2.append(kmeans_l2)
    
    print(f"Created {global_cluster_idx} total clusters")
    return cluster_assignments, global_cluster_idx

# Step 4: Create histogram representations
def create_histograms(data_list, segments, cluster_assignments, n_clusters, segment_length):
    """
    Create histogram representation for each original data sample.
    Returns a list of histograms and their corresponding activity labels.
    """
    if n_clusters == 0 or segments.size == 0:
        print("Error: No clusters or segments to create histograms!")
        return np.array([]), np.array([])
    
    # Group segments back to their original samples
    sample_histograms = []
    sample_activities = []
    
    segment_idx = 0
    unique_activities = set()
    
    for item in tqdm(data_list, desc="Creating histograms"):
        # Calculate how many segments this sample has
        n_segments = (len(item['data']) - segment_length) // segment_length
        
        if n_segments <= 0:
            continue
            
        # Get the cluster assignments for these segments
        if segment_idx + n_segments > len(cluster_assignments):
            print(f"Warning: Not enough cluster assignments ({len(cluster_assignments)}) for all segments")
            break
            
        sample_clusters = cluster_assignments[segment_idx:segment_idx+n_segments]
        
        # Create histogram
        hist, _ = np.histogram(sample_clusters, bins=n_clusters, range=(0, n_clusters))
        hist = hist / (np.sum(hist) + 1e-10)  # Normalize, avoid division by zero
        
        sample_histograms.append(hist)
        sample_activities.append(item['activity'])
        unique_activities.add(item['activity'])
        
        segment_idx += n_segments
    
    print(f"Created {len(sample_histograms)} histograms for {len(unique_activities)} activities")
    return np.array(sample_histograms), np.array(sample_activities)

# Step 5: Train a classifier
def train_classifier(histograms, activities):
    """
    Train a Random Forest classifier on the histogram representations.
    Returns the trained classifier and evaluation metrics.
    """
    if histograms.size == 0 or activities.size == 0:
        print("Error: No data to train classifier!")
        return None, 0, None, None, None
    
    if len(histograms) != len(activities):
        print(f"Error: Mismatch in histograms ({len(histograms)}) and activities ({len(activities)})")
        return None, 0, None, None, None
    
    # Need at least 2 samples per class for stratification
    unique_counts = {a: list(activities).count(a) for a in set(activities)}
    min_count = min(unique_counts.values())
    
    if min_count < 2:
        print(f"Warning: Some activities have less than 2 samples. Using regular train/test split.")
        stratify = None
    else:
        stratify = activities
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        histograms, activities, test_size=0.2, random_state=42, stratify=stratify
    )
    
    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return clf, accuracy, conf_matrix, y_test, y_pred

# Step 6: Visualization function
def plot_confusion_matrix(conf_matrix, class_names):
    """
    Plot the confusion matrix.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return plt.gcf()



# Main execution
if __name__ == "__main__":
    # Get data path from user input or use default
    default_path = "./HMP_Dataset"
    data_path = input(f"Enter path to dataset (or press Enter for default '{default_path}'): ") or default_path
    
    print(f"Using data path: {data_path}")
    
    # Define parameters
    segment_length = 32  # 1 second at 32 Hz
    level1_k = 40
    level2_k = 12
    
    # Step 1: Load data
    data_list = load_data(data_path)
    
    if not data_list:
        print("Error: No data loaded. Please check the path and data format.")
        exit(1)
    
    # Step 2: Segment data
    segments, segment_activities = segment_data(data_list, segment_length)
    
    if segments.size == 0:
        print("Error: No segments created. Cannot continue.")
        exit(1)
    
    # Step 3: Hierarchical K-means
    cluster_assignments, n_clusters = hierarchical_kmeans(segments, level1_k, level2_k)
    
    if n_clusters == 0:
        print("Error: Clustering failed. Cannot continue.")
        exit(1)
    
    # Step 4: Create histograms
    histograms, activities = create_histograms(data_list, segments, cluster_assignments, n_clusters, segment_length)
    
    if histograms.size == 0:
        print("Error: No histograms created. Cannot continue.")
        exit(1)
    
    # Step 5: Train classifier
    clf, accuracy, conf_matrix, y_test, y_pred = train_classifier(histograms, activities)
    
    if clf is None:
        print("Error: Classifier training failed. Cannot continue.")
        exit(1)
    
    # Print results
    print(f"\n--- Results with segment_length={segment_length}, level1_k={level1_k}, level2_k={level2_k} ---")
    print(f"Classification accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix
    unique_activities = np.unique(activities)
    cm_fig = plot_confusion_matrix(conf_matrix, unique_activities)
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # Save detailed results to file
    result_file = "classification_results.txt"
    with open(result_file, "w") as f:
        f.write(f"ADL Classification Results\n")
        f.write(f"=========================\n\n")
        f.write(f"Parameters:\n")
        f.write(f"- Segment length: {segment_length}\n")
        f.write(f"- Level 1 clusters: {level1_k}\n")
        f.write(f"- Level 2 clusters: {level2_k}\n")
        f.write(f"- Total clusters: {n_clusters}\n\n")
        
        f.write(f"Results:\n")
        f.write(f"- Accuracy: {accuracy:.4f}\n\n")
        
        f.write(f"Confusion Matrix:\n")
        cm_str = "\t" + "\t".join(unique_activities) + "\n"
        for i, activity in enumerate(unique_activities):
            cm_str += f"{activity}\t" + "\t".join([str(x) for x in conf_matrix[i]]) + "\n"
        f.write(cm_str)
    
    print(f"Detailed results saved to '{result_file}'")
   
    
    # Part (b): Experiment with different parameters
    run_parameter_tuning = input("Run parameter tuning experiments? (y/n): ").lower() == 'y'
    
    if run_parameter_tuning:
        print("\n--- Parameter Tuning ---\n")
        
        # Try different segment lengths
        segment_lengths = [16, 32, 64]
        # Try different cluster configurations
        cluster_configs = [(20, 6), (40, 12), (60, 18)]
        
        results = []
        
        for seg_len in segment_lengths:
            for level1, level2 in cluster_configs:
                print(f"\nTesting segment_length={seg_len}, level1_k={level1}, level2_k={level2}")
                
                # Repeat steps 2-5 with new parameters
                segments, segment_activities = segment_data(data_list, seg_len)
                
                if segments.size == 0:
                    print("  Skipping this configuration due to segmentation failure")
                    continue
                
                cluster_assignments, n_clusters = hierarchical_kmeans(segments, level1, level2)
                
                if n_clusters == 0:
                    print("  Skipping this configuration due to clustering failure")
                    continue
                
                histograms, activities = create_histograms(data_list, segments, cluster_assignments, n_clusters, seg_len)
                
                if histograms.size == 0:
                    print("  Skipping this configuration due to histogram creation failure")
                    continue
                
                _, accuracy, _, _, _ = train_classifier(histograms, activities)
                
                results.append({
                    'segment_length': seg_len,
                    'level1_k': level1,
                    'level2_k': level2,
                    'n_clusters': n_clusters,
                    'accuracy': accuracy
                })
                
                print(f"  Accuracy: {accuracy:.4f}")
        
        # Find best parameters
        if results:
            results_df = pd.DataFrame(results)
            best_result = results_df.loc[results_df['accuracy'].idxmax()]
            
            print("\n--- Best Parameters ---")
            print(f"Segment length: {best_result['segment_length']}")
            print(f"Level 1 clusters: {best_result['level1_k']}")
            print(f"Level 2 clusters: {best_result['level2_k']}")
            print(f"Total clusters: {best_result['n_clusters']}")
            print(f"Accuracy: {best_result['accuracy']:.4f}")
            
            # Save parameter tuning results
            results_df.to_csv("parameter_tuning_results.csv", index=False)
            print("Parameter tuning results saved to 'parameter_tuning_results.csv'")
        else:
            print("No valid results from parameter tuning")
            