function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


num_theta = size(theta)(1);				% number of rows within the vector
sig_input = X * theta;					% results of theta transpose * X that feeds into sigmoid


% compute cost
cost_true = log(sigmoid(sig_input));
cost_false = log(1 - sigmoid(sig_input));
costs = (-y .* cost_true) - ((1-y) .* cost_false);
sq_thetas = theta.^2;
J = (1/m) * ( sum(costs) + (lambda/2) * sum(sq_thetas(2:num_theta,:)) );  % do not regularize theta_0 


% compute gradient
for i = 1:num_theta
	if (i == 1)
		grad(i,1) = (1/m) * sum((sigmoid(sig_input)-y) .* X(:,i));
	else
		grad(i,1) = (1/m) * sum((sigmoid(sig_input)-y) .* X(:,i)) + (lambda/m) * theta(i,1);
end

%fprintf('grad: \n');
%fprintf(' %f \n', grad);

% =============================================================

end
