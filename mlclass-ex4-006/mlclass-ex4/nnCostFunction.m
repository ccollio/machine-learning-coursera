function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Add ones to the X data matrix
X = [ones(m, 1) X];

total_cost = 0;
cost_k = 0;

% for each training example
for i = 1:m

	% compute hidden layer
	a_1 = X(i,:);
	z_2 = a_1 * Theta1';		% 1x401 * 401x25
	a_2 = sigmoid(z_2);			% 1x25

	% compute output layer
	a_2 = [1 a_2];				% 1x26
	z_3 = a_2 * Theta2'; 		% 1x26 * 26x10
	a_3 = sigmoid(z_3);			% 1x10
	a_3 = a_3';					% 10x1


	% set up the y vector 
	y_k = zeros(num_labels,1);
	y_k(y(i,1),1) = 1;

	% compute the cost
	cost_true = log(a_3);
	cost_false = log(1 - a_3);
	cost_k = (-y_k .* cost_true) - ((1-y_k) .* cost_false);

	% add on the costs from this example
	total_cost = total_cost + sum(cost_k);

end


% compute regularization inputs
Theta1_no_bias = Theta1(:,2:input_layer_size+1);
Theta1_sq = Theta1_no_bias.^2;

Theta2_no_bias = Theta2(:,2:hidden_layer_size+1);
Theta2_sq = Theta2_no_bias.^2;

cost_reg = (1/2) * (lambda/m) * (sum(Theta1_sq(:)) + sum(Theta2_sq(:)));


% return cost across all m examples
J = (1/m) * total_cost + cost_reg;


% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Delta_1 = zeros(size(Theta1_grad));
Delta_2 = zeros(size(Theta2_grad));


% for each training example
for t = 1:m

	% =============== FORWARD PASS ===============

	% compute hidden layer activations
	a_1 = X(t,:);				% 1x401
	z_2 = a_1 * Theta1';		% 1x401 * 401x25 = 1x25
	a_2 = sigmoid(z_2);			% 1x25

	% compute output layer activations
	a_2 = [1 a_2];				% 1x26
	z_3 = a_2 * Theta2'; 		% 1x26 * 26x10
	a_3 = sigmoid(z_3);			% 1x10
	a_3 = a_3';					% 10x1

	% =============== BACKPROPAGATION ===============

	% set up the y vector 
	y_j = zeros(num_labels,1);
	y_j(y(t,1),1) = 1;

	% compute output layer errors
	d_3 = a_3 - y_j;						% 10x1

	% compute hidden layer errors
	d_2_input = Theta2' * d_3;
	d_2 =  d_2_input(2:end) .* sigmoidGradient(z_2');  	% 25x1 .* 25x1  


	% accumulate the gradients
	Delta_1 = Delta_1 + d_2 * a_1;  		% 25x1 * 1x401
	Delta_2 = Delta_2 + d_3 * a_2;			% 10x1 * 1x26

end

Theta1_grad = (1/m) .* Delta_1;
Theta2_grad = (1/m) .* Delta_2;


% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% compute regularization inputs
% don't regularize the first column of the Thetas
Theta_1_grad_reg = (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
Theta_2_grad_reg = (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];

Theta1_grad = Theta1_grad + Theta_1_grad_reg;
Theta2_grad = Theta2_grad + Theta_2_grad_reg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
