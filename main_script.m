clear all;
clc;

% Step 1: Load the training data
data = load('BD1.txt'); % Binary vectors of length 25
NP = 25; % Number of pixels per vector
K = 3; % Number of clusters
max_iterations = 100; % Maximum number of iterations

% Step 2: Perform Lloyd's Clustering / choose the function
%[cluster_centers, cluster_assignments] = lloyd_clustering_kmeanspp(data, K, NP, max_iterations);
[cluster_centers, cluster_assignments] = lloyd_clustering(data, K, NP, max_iterations);

% Step 3: Load test datasets
test1 = load('teste.txt');
test2 = load('teste2.txt');

% Split features and labels
test1_features = test1(:, 1:NP);
test1_labels = test1(:, NP + 1);

test2_features = test2(:, 1:NP);
test2_labels = test2(:, NP + 1);

% Step 4: Evaluate the classifier on both test datasets
disp('Results for Test1:');
evaluate_classifier(test1_features, test1_labels, cluster_centers, K, NP);

disp('Results for Test2:');
evaluate_classifier(test2_features, test2_labels, cluster_centers, K, NP);
