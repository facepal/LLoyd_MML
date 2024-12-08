clear all;
clc;

% Parameters
data = load('BD1.txt'); % Training dataset
test1 = load('teste.txt'); % Test dataset 1
test2 = load('teste2.txt'); % Test dataset 2

NP = 25; % Number of pixels per vector
K = 3; % Number of clusters
max_iterations = 100; % Max iterations for Lloyd's algorithm
num_simulations = 500; % Number of Monte Carlo simulations

% Extract features and labels for test datasets
test1_features = test1(:, 1:NP);
test1_labels = test1(:, NP + 1);

test2_features = test2(:, 1:NP);
test2_labels = test2(:, NP + 1);

% Initialize storage for metrics
results_random = zeros(num_simulations, 4); % Accuracy, Precision, Recall, F1 for random
results_kmeanspp = zeros(num_simulations, 4); % Accuracy, Precision, Recall, F1 for k-means++

% Monte Carlo Simulation
for sim = 1:num_simulations
    disp(['Simulation ', num2str(sim), ' of ', num2str(num_simulations)]);

    % Run Lloyd's clustering with random initialization
    [cluster_centers_random, ~] = lloyd_clustering(data, K, NP, max_iterations);
    metrics_random_test1 = evaluate_classifier_mc(test1_features, test1_labels, cluster_centers_random, K, NP);
    metrics_random_test2 = evaluate_classifier_mc(test2_features, test2_labels, cluster_centers_random, K, NP);

    % Average the metrics for Test1 and Test2
    results_random(sim, :) = mean([metrics_random_test1; metrics_random_test2]);

    % Run Lloyd's clustering with K-Means++ initialization
    [cluster_centers_kmeanspp, ~] = lloyd_clustering_kmeanspp(data, K, NP, max_iterations);
    metrics_kmeanspp_test1 = evaluate_classifier_mc(test1_features, test1_labels, cluster_centers_kmeanspp, K, NP);
    metrics_kmeanspp_test2 = evaluate_classifier_mc(test2_features, test2_labels, cluster_centers_kmeanspp, K, NP);

    % Average the metrics for Test1 and Test2
    results_kmeanspp(sim, :) = mean([metrics_kmeanspp_test1; metrics_kmeanspp_test2]);
end



%% Displaying results

% Aggregate results
average_random = mean(results_random);
std_random = std(results_random);
max_random = max(results_random);

average_kmeanspp = mean(results_kmeanspp);
std_kmeanspp = std(results_kmeanspp);
max_kmeanspp = max(results_kmeanspp);

% Display results
disp('Monte Carlo Simulation Results:');
disp('Random Initialization:');
fprintf('Accuracy: %.2f%% ± %.2f%% (Max: %.2f%%)\n', average_random(1) * 100, 2 * std_random(1) * 100, max_random(1) * 100);
fprintf('Precision: %.2f%% ± %.2f%% (Max: %.2f%%)\n', average_random(2) * 100, 2 * std_random(2) * 100, max_random(2) * 100);
fprintf('Recall: %.2f%% ± %.2f%% (Max: %.2f%%)\n', average_random(3) * 100, 2 * std_random(3) * 100, max_random(3) * 100);
fprintf('F1-Score: %.2f%% ± %.2f%% (Max: %.2f%%)\n', average_random(4) * 100, 2 * std_random(4) * 100, max_random(4) * 100);

disp('K-Means++ Initialization:');
fprintf('Accuracy: %.2f%% ± %.2f%% (Max: %.2f%%)\n', average_kmeanspp(1) * 100, 2 * std_kmeanspp(1) * 100, max_kmeanspp(1) * 100);
fprintf('Precision: %.2f%% ± %.2f%% (Max: %.2f%%)\n', average_kmeanspp(2) * 100, 2 * std_kmeanspp(2) * 100, max_kmeanspp(2) * 100);
fprintf('Recall: %.2f%% ± %.2f%% (Max: %.2f%%)\n', average_kmeanspp(3) * 100, 2 * std_kmeanspp(3) * 100, max_kmeanspp(3) * 100);
fprintf('F1-Score: %.2f%% ± %.2f%% (Max: %.2f%%)\n', average_kmeanspp(4) * 100, 2 * std_kmeanspp(4) * 100, max_kmeanspp(4) * 100);

%% Plotting

metrics = {'Accuracy', 'Precision', 'Recall', 'F1-Score'};
x = categorical(metrics);
x = reordercats(x, metrics);

% Bar Chart Data
mean_data = [average_random; average_kmeanspp]' * 100; % Convert to percentages
std_data = [std_random; std_kmeanspp]' * 100; % Convert to percentages

% Plot Bar Chart
figure;
hold on;
bar_width = 0.8; % Width of bars
b = bar(x, mean_data, bar_width); % Store bar handles

% Center error bars on the bars
for i = 1:length(b)
    % Use XData and XOffset to correctly position error bars
    x_data = b(i).XEndPoints; % Use the XEndPoints property for correct alignment
    errorbar(x_data, mean_data(:, i), std_data(:, i), 'k.', 'LineWidth', 1.5); % Add error bars
end

% Add legend, labels, and adjust title spacing
legend('Random Initialization', 'K-Means++ Initialization', 'Location', 'northwest');
ylabel('Percentage (%)');
title('Comparison of Random vs K-Means++ Initialization');
set(gca, 'TitleFontSizeMultiplier', 1.2); % Increase space for title
ylim([0, max(mean_data(:) + std_data(:)) + 10]); % Add extra space above bars for clarity

hold off;


