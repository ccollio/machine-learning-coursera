function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

max_val = 0;
max_pos = 0;
classifier_output = zeros(1, num_labels);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% for each training example
for i = 1:m

	% compute hidden layer
	z_2 = X(i,:) * Theta1';    	% 1 X 401  *  401 X 25
	a_2 = sigmoid(z_2);  	 	% 1 X 25


	% compute ouput layer
	a_2 = [1 a_2];
	z_3 = a_2 * Theta2';    % 1 X 26  *	26 X 10
	a_3 = sigmoid(z_3);  	% 1 X 10


	% find the classifier that had the max probability
 	[max_val, max_pos] = max(a_3);
 	p(i,1) = max_pos;

end


% =========================================================================


end
