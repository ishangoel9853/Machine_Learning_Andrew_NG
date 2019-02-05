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

h_x = X * theta ;  % (12*2)*(2*1)

temp1 = h_x - y ;   % 12*1

temp2 = sum( theta .^ 2 ) - theta(1)^2 ;

J= (1/(2*m)) * sum( temp1 .^ 2 )  +  (lambda / (2*m) ) * temp2;


%------GRADIENT : 

temp3 = (1/m)*(temp1' * X)' + (lambda/m) * theta ;

grad(2:end) = temp3(2:end) ;
grad(1) = (1/m) * (temp1' * X)(1);



% =========================================================================

grad = grad(:);

end
