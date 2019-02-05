function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.01;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
%{
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    predictions= svmPredict(model,Xval);
    err= mean(double(predictions -=yval));

for a = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
  for b = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    model= svmTrain(X, y, a, @(x1, x2) gaussianKernel(x1, x2, b)); 
    predictions= svmPredict(model,Xval);
    errn= mean(double(predictions ~=yval));
    if(errn<err)
      err=errn;
      temp1=a
      temp2=b
    endif
  end
end

C=temp1
sigma=temp2
%}

C=1;
sigma =  0.1;






% =========================================================================

end
