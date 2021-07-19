% assume the first 2 columns are methods to be compared

number_of_methods =6;
filename = 'results_convertlean.csv';

data_table = readtable(filename);
data_internal = data_table{:, 2:end-1};                 % curly bracket for data extraction
problems = data_table{:, 1};                        % extract problem name 
methods = data_table.Properties.VariableNames(2:end-1); % read table head


results = csvread('hvraw.csv');

n = size(results, 2)/number_of_methods;     % infer number of problems

% prepare output file
fid = fopen('hvcompare_sig.csv', 'w');
% write header, assume the first 2 columns are algorithm to be compared
fprintf(fid, 'method-problem,');
fprintf(fid, methods{1});
fprintf(fid, ',');
fprintf(fid, methods{2});
fprintf(fid, ',');
% write the rest header
for i = 3: number_of_methods
    fprintf(fid, methods{i});
    fprintf(fid, ',');
    fprintf(fid, 'test_to_1st_method,');
    fprintf(fid, 'test_to_2nd_method,');
end
fprintf(fid, '\n');
% fclose(fid);

for i = 1:n % process each problem 
    % print problem name 
    fprintf(fid, problems{i});
    fprintf(fid, '& ,');
    fprintf(fid,  '%.4f &, ',data_internal(i, 1));
    fprintf(fid,  '%.4f  , ',data_internal(i, 2));
    
    for j = 3: number_of_methods % process each method, compare to the first 2
        fprintf(fid,  '& %.4f, ', data_internal(i, j));
        
        current_method = results(:, (i-1) * number_of_methods + j);
        
        for k = 1: 2
            method1 = results(:, (i-1)*number_of_methods + k);
            % 1 means  the former is smaller,  second is bigger/better, 
            % for hv comparison larger the better
            [p1,h1,stats1] = ranksum(method1, current_method,  'alpha', 0.05, 'tail', 'left');           
            [p2,h2,stats2] = ranksum(current_method, method1,  'alpha', 0.05, 'tail', 'left'); 
            
            if h1 == 1 && h2 == 0
               fprintf(fid, '$\\uparrow_{%d}$, ', k);
            elseif h2==1 && h1 == 0
               fprintf(fid, '$\\downarrow_{%d}$, ', k);
            else
                 fprintf(fid, '$\\approx_{%d}$, ', k);
            end  
            
          
        end      
    end
    fprintf(fid, '\\');
    fprintf(fid, '\\');
    fprintf(fid, '\n');
end
fclose(fid);



