function [cluster_centers, cluster_assignments] = lloyd_clustering_kmeanspp_dice(data, K, NP, max_iterations)
    % Input:
    % data: NxD matrix (N samples, D=NP features)
    % K: Number of clusters
    % NP: Number of pixels in each binary vector
    % max_iterations: Maximum number of iterations for convergence
    % Output:
    % cluster_centers: KxD matrix (K cluster centers)
    % cluster_assignments: Nx1 vector of cluster assignments for each data point

    % Step 1: K-means++ initialization
    N = size(data, 1);
    cluster_centers = zeros(K, NP); % KxD matrix for cluster centers

    % Select the first cluster center randomly
    %rng(0); % Set seed for reproducibility
    cluster_centers(1, :) = data(randi(N), :);

    % Select the remaining K-1 cluster centers
    for k = 2:K
        % Compute the distance of each point to the nearest cluster center
        distances = inf(N, 1);
        for j = 1:k-1
            intersection = sum(data & cluster_centers(j, :), 2);
            sum_data_cluster = sum(data, 2) + sum(cluster_centers(j, :));
            dice_dissimilarity = 1 - (2 * intersection ./ sum_data_cluster);
            distances = min(distances, dice_dissimilarity);
        end
        % Probability proportional to the squared distance
        probabilities = distances / sum(distances);
        cumulative_probabilities = cumsum(probabilities);
        r = rand();
        selected_index = find(cumulative_probabilities >= r, 1);
        cluster_centers(k, :) = data(selected_index, :);
    end

    % Step 2: Lloyd's Algorithm
    for iteration = 1:max_iterations
        % Assignment step: Assign each point to the nearest cluster
        cluster_assignments = zeros(N, 1);
        for i = 1:N
            distances = zeros(K, 1);
            for k = 1:K
                intersection = sum(data(i, :) & cluster_centers(k, :));
                sum_data_cluster = sum(data(i, :)) + sum(cluster_centers(k, :));
                distances(k) = 1 - (2 * intersection / sum_data_cluster); % Dice dissimilarity
            end
            [~, cluster_assignments(i)] = min(distances); % Assign to the nearest cluster
        end

        % Update step: Calculate the medoid for each cluster
        new_cluster_centers = cluster_centers;
        for k = 1:K
            cluster_points = data(cluster_assignments == k, :); % Points in cluster k
            if isempty(cluster_points)
                continue; % Skip empty clusters
            end
            % Find the medoid (minimize intra-cluster distance)
            min_cost = inf;
            for i = 1:size(cluster_points, 1)
                intersection_matrix = cluster_points & cluster_points(i, :);
                sum_matrix = sum(cluster_points, 2) + sum(cluster_points(i, :));
                cost = sum(1 - (2 * sum(intersection_matrix, 2) ./ sum_matrix));
                if cost < min_cost
                    min_cost = cost;
                    new_cluster_centers(k, :) = cluster_points(i, :);
                end
            end
        end

        % Convergence check: Stop if cluster centers do not change
        if isequal(new_cluster_centers, cluster_centers)
            disp(['Converged in ', num2str(iteration), ' iterations.']);
            break;
        end

        % Update cluster centers
        cluster_centers = new_cluster_centers;
    end
end
