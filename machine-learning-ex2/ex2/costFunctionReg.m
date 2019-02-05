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

temp1 = X * theta ; % temp1 - 100x1
hox = sigmoid(temp1) ; % hox - 100x1

n=length(theta);

tempf = - (y .* log(hox)) - ((1-y) .* log(1-hox)); % tempf- 100x1
tempf2= sum(theta .^ 2) - (theta(1)*theta(1));


J=( (sum(tempf))/m )+( (lambda/(2*m))*tempf2 ) ;

grad = ( ( X' * (hox - y) ) ./ m ) + ((lambda * theta) ./ m );
grad(1) = ( X'(1,:) * (hox - y) ) ./ m;




% =============================================================

end
