function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

num_theta = size(theta)(1);				% number of rows within the vector
sig_input = X * theta;					% results of theta transpose * X that feeds into sigmoid


% compute cost
cost_true = log(sigmoid(sig_input));
cost_false = log(1 - sigmoid(sig_input));
costs = (-y .* cost_true) - ((1-y) .* cost_false);
J = (1/m) * sum(costs);

% better implementation of cost function
% J = 1./m * ( -y' * log( sigmoid(X * theta) ) - ( 1 - y' ) * log ( 1 - sigmoid( X * theta)) );


% compute gradient
for i = 1:num_theta

	grad(i,1) = (1/m) * sum((sigmoid(sig_input)-y) .* X(:,i));

end

% better implementation to compute gradient
%grad = 1./m * X' * (sigmoid(sig_input) - y);


% =============================================================

end
