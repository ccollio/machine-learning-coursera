function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% container for parameter values to try based on initial C and sigma
param_values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
num_params = length(param_values);

% container for C, sigma, and error values
param_errors = zeros(num_params^2, 3);


count = 1;
for i = 1:num_params

	C_current = param_values(i);
	for j = 1:num_params

		sigma_current = param_values(j);
		model = svmTrain(X, y, C_current, @(x1, x2) gaussianKernel(x1, x2, sigma_current));
		predictions = svmPredict(model, Xval);

		% store C, sigma and prediction error
		param_errors(count,1) = C_current;
		param_errors(count,2) = sigma_current;
		param_errors(count,3) = mean(double(predictions ~= yval));

		count++;
	end
end

% find index of minimum prediction error
[min_val, min_index] = min(param_errors(:,3))

% return the C and sigma values
C = param_errors(min_index,1);
sigma = param_errors(min_index,2);

% =========================================================================

end
