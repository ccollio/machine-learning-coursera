
function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% ~~~~~~~~~COMPUTE THE COST~~~~~~~~~

num_theta = size(theta)(1);				% capture num rows of theta
predictions = X * theta;				% predictions of hypothesis on all m examples
sqrErrors = (predictions - y).^2;		% square sqrErrors
sq_thetas = theta.^2;					% square theta values

% return the result of the regularized cost function
J = 1/(2 * m) * sum(sqrErrors) + (lambda/(2*m) * sum(sq_thetas(2:num_theta,:)));			
								

% ~~~~~~~~~COMPUTE THE GRADIENT~~~~~~~~~

mask = ones(size(theta));		
mask(1) = 0;							% no regularization of theta_0

grad = (1/m) .* (X' * (predictions-y)) + (lambda/m) .* (theta .* mask)


% =========================================================================

grad = grad(:);

end
