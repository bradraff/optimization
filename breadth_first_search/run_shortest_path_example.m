% % Show an example 10x10 random binary matrix; this matrix is used as a
% directed graph; each row is a node and each unity value is an edge from
% the row in which that unity value exists to the node represented by the
% column location of that unity value.

A = randBinMatrix(10, 10); % Generate a random binary matrix of size (10,10)
disp('An example random binary 10x10 matrix:')
disp(A); % An example random matrix

% % Execute BFS on this one example matrix
[path, err] = find_shortest_path(A, 1, 10); % Find least-number-of-hops path from node 1 to node 10 for matrix A; if such a path does not exist, error = 1
disp('Least-hops-path for the above matrix, from node 1 to node 10, is:')
disp(path)

% % Execute BFS on 1000 random binary matrices
for i = 1:1000
    A = randBinMatrix(10, 10); 
    [path, err] = find_shortest_path(A, 1, 10);
    numHops(i)  = length(path) - 1; % Number of hops for case i = (#nodes along path) - 1
    errLog(i)   = err;
end
% % Count number of errors
numErrs = sum(errLog);

% % Remove the errorcases from the numHops array
numHops(errLog == 1) = []; 

% % Determine mean number of hops across all 1000 cases
meanNumHops = mean(numHops);

% % Display results to command window
fprintf('The number of errors for this batch of 1000 random binary matrices was %d\n', numErrs)
fprintf('The percent of feasible paths was therefore %0.2f%%\n', (1 - numErrs/1000) * 100)
fprintf('The mean number of hops required to reach node 10 from node 1 was %0.3f\n', meanNumHops)

%% Functions
% % Generate a random binary matrix
function A  = randBinMatrix(m, n)
    A   = zeros(m,n);
    for i = 1:m
        for j = 1:n
            A(i,j) = randi([0 1]); % ith row and jth column is a random integer 0 or 1
        end
    end
end
