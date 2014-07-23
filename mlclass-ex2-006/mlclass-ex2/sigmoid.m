function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% capture the number of rows and columns
[r c] = size(z);

% loop through each element of the input parameter
for i = 1:r

	for j = 1:c

		% run the sigmoid function 
		g(i,j) = 1 / (1 + exp(-1 * z(i,j)));

	end

end



% =============================================================

end
