function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
% You need to return the following variables correctly 
%{
data= load('ex1data1.txt');

y=data(:,2);
X=[ones(m,1) data(:,1)];



theta = zeros(2,1);
%}
J = 0;
m = length(y); 

%iterations = 1500;
%alpha = 0.02;

%for i=1:iterations
%theta =  theta - (alpha/(2*m)) * sum( ((X * theta) - y)' * X , 1 )';
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

J= sum(((X*theta)-y) .^ 2 ) /(2*m);



% =========================================================================

end
