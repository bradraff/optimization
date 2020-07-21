function [top_sort, is_cyclic] = topological_sort(G)
%%topological_sort is an implementation of Kahn's Algorithm: A.B. Kahn,
%%"Topological Sorting of Large Networks," Communications of the ACM, vol.
%%5, no. 11, pp. 558-562, 1962. This implementation will not only sort an
%%acyclic dependency graph, but it will also identify cyclic dependency
%%graphs. The input, G, is binary matrix whose rows represent nodes and
%%whose unity values represent edges between nodes. The first output is the
%%proper topological sort of the graph, or, if the graph is cyclic, it will
%%report so to the command window. The second output is a boolean that is 1
%%if the graph is cyclic and 0 if the graph is acyclic.
%
%
% % Example input graph:
% G =   [0 0 1;
%        1 0 0;
%        0 0 0]
% % G's first node has an edge to the third node, and its second node has an
% % edge to the first.
%
% % Example using above G
%
% [top_sort, is_cyclic] = topological_sort(G)
% % Returns:
% top_sort =
%   2   1   3
% is_cyclic = 
%   0

%% Compute indegree for each vertex in the graph (number of incoming edges for a given node)
[num_rows, num_cols]  = size(G);
num_nodes            = num_rows; % Number of nodes in graph = number of rows

for i = 1:num_cols
    incoming_edges 	= find(G(:,i));             % Vertices whose edges go into to node i
    indegree(i)     = length(incoming_edges);    % Number of incoming edges to node i
end

%% Identify nodes who have no incoming edges (indegree = 0)
queue   = find(indegree == 0);

%% Sort dependency graph
cnt     = 0;            % Initialize counter for while loop
top_sort   = [];           % The sorted topological order of the graph; algorithm will fill in
while ~isempty(queue)   % While S is not empty

    n               = queue(1);     % Place node n at the front of the queue
    queue           = queue(2:end); % Remove node n from the queue
    top_sort(end+1)    = n;            % Append node n to the order     

    % Find nodes who receive edge from node n, as their indegree
    % will decrease by one once node n is reomved from the graph. Call
    % these nodes "neighboring" nodes
    neighbors = find(G(n, :));   % Examine all columns in nth row of G (nth node of G)
    for i = 1:length(neighbors)  % For each neighboring node
        indegree(neighbors(i)) = indegree(neighbors(i)) - 1; % Decrease indegree of each neighboring node by 1
        if indegree(neighbors(i)) == 0   % If the indegree of a neighboring node now equals 0
            queue(end+1) = neighbors(i); % Add the neighboring node to the queue 
        end
    end

    cnt = cnt + 1;  % Increment count of visited nodes

end

%% Print the result of analyzing the dependnecy graph
if cnt == num_nodes % If the number of visited nodes = the number of total nodes
    is_cyclic = 0;
    fprintf('\nThe sorted topological order for the graph is:\n')
    disp(top_sort)
    fprintf('\t. . . where each number represents a node (row) from the original graph.\n')
else % If the number of visited nodes ~= the number of total nodes
    is_cyclic = 1;
    fprintf('\nGraph is cyclic!\n\n')
end

end % function