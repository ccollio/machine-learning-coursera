function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%     


f_count = 1;									% initialize the feature counter
num_features = size(X,2);						% initialize the number of iterations
for iter = 1:num_features

	mu(1,f_count) = mean(X(:,f_count));
	sigma(1,f_count) = std(X(:,f_count));

	f_count++;
end


f_count = 1;									% initialize the feature counter
ex_count = 1;									% initialize the example counter
num_examples = size(X,1);						% initialize the number of iterations
for iter = 1:num_examples

	% normalize each feature in the example
	for iter = 1:num_features
		
		X_norm(ex_count,f_count) = (X(ex_count,f_count) - mu(1,f_count)) / sigma(1,f_count);

		f_count++;
	end

	f_count = 1;								% reinitialize the feature counter for next example
	ex_count++;
end









% ============================================================

end
