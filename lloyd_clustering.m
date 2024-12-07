function [cluster_centers, cluster_assignments] = lloyd_clustering(data, K, NP, max_iterations)
    % Input:
    % data: NxD matrix (N samples, D=NP features)
    % K: Number of clusters
    % NP: Number of pixels in each binary vector
    % max_iterations: Maximum number of iterations for convergence
    % Output:
    % cluster_centers: KxD matrix (K cluster centers)
    % cluster_assignments: Nx1 vector of cluster assignments for each data point

    % Step 1: Initialize cluster centers randomly
    %rng(0); % Set seed for reproducibility
    N = size(data, 1); % Number of data points
    cluster_centers = data(randperm(N, K), :); % Randomly pick K points as initial centers

    % Step 2: Lloyd's algorithm
    for iteration = 1:max_iterations
        % Assignment step: Assign each point to the nearest cluster
        cluster_assignments = zeros(N, 1);
        for i = 1:N
            distances = sum(abs(data(i, :) - cluster_centers), 2) / NP; % Normalized Hamming
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
                cost = sum(sum(abs(cluster_points - cluster_points(i, :)), 2) / NP);
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
