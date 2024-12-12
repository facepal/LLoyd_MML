clear all;
clc;

% Step 1: Load the training data
data = load('BD1.txt');
NP = 25; % Number of pixels
K = 3; % Number of clusters
max_iterations = 100; % Maximum number of iterations

% Step 2: Perform Lloyd's Clustering with Random Initialization
disp('Running Lloyd Clustering with Random Initialization...');
[cluster_centers_random, ~] = lloyd_clustering_jacc(data, K, NP, max_iterations);

% Step 3: Perform Lloyd's Clustering with K-Means++ Initialization
disp('Running Lloyd Clustering with K-Means++ Initialization...');
[cluster_centers_kmeanspp, ~] = lloyd_clustering_kmeanspp_jacc(data, K, NP, max_iterations);

% Step 4: Load test datasets
test1 = load('teste.txt');
test2 = load('teste2.txt');

test1_features = test1(:, 1:NP);
test1_labels = test1(:, NP + 1);

test2_features = test2(:, 1:NP);
test2_labels = test2(:, NP + 1);

% Step 5: Evaluate the classifier with Random Initialization
disp('Evaluating Lloyd Clustering with Random Initialization on Test1:');
evaluate_classifier_jacc(test1_features, test1_labels, cluster_centers_random, K, NP);

disp('Evaluating Lloyd Clustering with Random Initialization on Test2:');
evaluate_classifier_jacc(test2_features, test2_labels, cluster_centers_random, K, NP);

% Step 6: Evaluate the classifier with K-Means++ Initialization
disp('Evaluating Lloyd Clustering with K-Means++ Initialization on Test1:');
evaluate_classifier_jacc(test1_features, test1_labels, cluster_centers_kmeanspp, K, NP);

disp('Evaluating Lloyd Clustering with K-Means++ Initialization on Test2:');
evaluate_classifier_jacc(test2_features, test2_labels, cluster_centers_kmeanspp, K, NP);
