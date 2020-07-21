function [path, err] = find_shortest_path(A, source, destination)
%find_shortest_path takes in matrix A, start node, source, and end node,
% destination. It returns the shortest-number-of-hops path from source to
% destination for a. Or, if no path exists, it returns an error. This is a
% Breadth-First-Search algorithm, first invented by Konrad Zuse in 1945.
% This particular implementation accepts directed, potentially-cyclic
% graphs and outputs the least-hops path.

%% Execute BFS Algorithm with Path Tracking
err = 0; % Initialize error as 0 (false, no error)
if isempty(find(A(1,:) , 1)) || isempty(find(A(1,:) , 1)) % If there are no edges to node 1 or node 10...
    err     = 1;    % ... then we are certain no path exists. Throw error.
    path    = [];   % The path is empty
    return          % Break out of function
end
    
[rows, ~]   = size(A);
numNodes  	= rows;         % Each row represents a node
nodes       = 1:numNodes;   % Array of nodes

for i = 1:numNodes          % For all nodes
    visited{i}  = 'false';  % Initialize ith node as not visited
    parent(i)   = -1;   	% Initialize ith previous/"parent" node to -1
end

queue           = [];           % Initialize queue of nodes to inspect  
visited{source} = 'true';       % Visit the source node
queue(end+1)	= source;       % Add source node to the queue
while queue % While queue is not empty
    n           = queue(1); % Select node n as first in queue
    queue(1)    = [];       % Dequeue node n
    
    if n == destination     % If we reach our destination
        break               % Break out of the while loop
    end
    
    neighbors = find(A(n,:));   % All nodes that neighbor node n, i.e. the column indices in row n with nonzero values
    for i = 1:length(neighbors) % For all neighboring nodes
            if strcmp(visited{neighbors(i)}, 'false') == 1 % Check if ith neighboring node has been visited; if not, proceed to below operations
                visited{neighbors(i)}	= 'true';   % Mark ith neighboring node as now being visited
                parent(neighbors(i))    = n;    	% Record node from where we came
                queue(end+1) = neighbors(i);        % Add ith neighboring node to the queue     
            end
    end
end

%% Trace path by backtracking from destination to source using the node history stored in path
currNode    = destination;
cnt = 1; % Initialize counter for the while loop
while 1 % This will run until no path is found or the current node is the source
    if (currNode == -1)
        err = 1;    % No path found, report error and break
        path = [];  % Path is empty since none exists
        break
    end
  	path(cnt)  = currNode;          % Build the path of nodes (in backwards order)
   	currNode = parent(currNode);    % The parent of the current node is now the current node
  	
    if currNode == source       % If the current node is the source
        path = [path, source];  % Append the source to the path
        path = flip(path);      % Reorder so that the path goes 1->10 instead of 10->1
        break
    end
    cnt = cnt+1;

end % while

end % bfs