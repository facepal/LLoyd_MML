function evaluate_classifier(test_features, test_labels, cluster_centers, K, NP)
    % Input:
    % test_features: NxD matrix of test data
    % test_labels: Nx1 vector of true labels
    % cluster_centers: KxD matrix of cluster centers
    % K: Number of clusters
    % NP: Number of pixels in each binary vector

    % Step 1: Classify test samples
    N = size(test_features, 1);
    predicted_labels = zeros(N, 1);
    for i = 1:N
        distances = sum(abs(test_features(i, :) - cluster_centers), 2) / NP; % Normalized Hamming
        [~, predicted_labels(i)] = min(distances); % Assign to the nearest cluster
    end

    % Step 2: Align clusters with true labels
    label_mapping = zeros(1, K); % Map clusters to true labels
    for cluster = 1:K
        cluster_indices = find(predicted_labels == cluster);
        if ~isempty(cluster_indices)
            label_mapping(cluster) = mode(test_labels(cluster_indices)); % Most common true label
        end
    end
    aligned_predictions = arrayfun(@(x) label_mapping(x), predicted_labels);

    % Step 3: Compute confusion matrix
    confusion_matrix = @(true_labels, predicted_labels) accumarray(...
        [true_labels, predicted_labels], 1, [K, K], @sum, 0);
    conf_matrix = confusion_matrix(test_labels, aligned_predictions);
    disp('Confusion Matrix:');
    disp(conf_matrix);

    % Step 4: Compute Metrics
    true_positives = diag(conf_matrix); % Diagonal values are true positives
    false_positives = sum(conf_matrix, 1)' - true_positives; % Column sums - TP
    false_negatives = sum(conf_matrix, 2) - true_positives; % Row sums - TP
    true_negatives = sum(conf_matrix(:)) - (false_positives + false_negatives + true_positives); % Total - FP - FN - TP

    % Precision, Recall, Specificity, F1-score
    precision = true_positives ./ (true_positives + false_positives);
    recall = true_positives ./ (true_positives + false_negatives); % Also called sensitivity
    specificity = true_negatives ./ (true_negatives + false_positives);
    f1_score = 2 * (precision .* recall) ./ (precision + recall);

    % Overall metrics
    accuracy = sum(true_positives) / sum(conf_matrix(:));
    macro_precision = mean(precision, 'omitnan'); % Average precision
    macro_recall = mean(recall, 'omitnan'); % Average recall
    macro_f1_score = mean(f1_score, 'omitnan'); % Average F1-score

    % Display Metrics
    disp('Metrics for Each Class:');
    for i = 1:K
        fprintf('Class %d: Precision = %.2f, Recall = %.2f, Specificity = %.2f, F1-Score = %.2f\n', ...
            i, precision(i) * 100, recall(i) * 100, specificity(i) * 100, f1_score(i) * 100);
    end

    fprintf('Overall Metrics: Accuracy = %.2f%%, Macro Precision = %.2f%%, Macro Recall = %.2f%%, Macro F1-Score = %.2f%%\n', ...
        accuracy * 100, macro_precision * 100, macro_recall * 100, macro_f1_score * 100);
end
