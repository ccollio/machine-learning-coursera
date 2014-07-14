function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    hypo = X * theta;                       % compute hypothesis for all examples
    

    theta_tmp = theta;                      % container for storing new theta values
    num_features = size(X,2);               % number of features
    f_count = 1;                            % keeps track current feature; initialize to theta_0

    for iter = 1:num_features

        if (f_count == 1)

            % special case for theta_0 because it is not multiplied by an x-value
            theta_tmp(f_count) = theta(f_count) - alpha * (1/m) * sum(hypo-y);

        else

            theta_tmp(f_count) = theta(f_count) - alpha * (1/m) * sum((hypo-y) .* X(:,f_count));

        endif

        f_count++;
    end


    % assign new theta values
    f_count = 1;
    for iter = 1:num_features

        theta(f_count) = theta_tmp(f_count);
        f_count++;

    end


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
