% Loop through each graph provided in the data folder and run
% topological_sort
for i = 1:4
    % % Set the graph of interest
    graph_file  = sprintf('./data/graph%d.csv', i);
    % % Load file
    G           = load(graph_file);
    % % Run topological_sort
    [top_sort, is_cyclic] = topological_sort(G);    
end