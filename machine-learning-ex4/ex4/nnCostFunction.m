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
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


a1 = [ones(m,1) X] ;  % 5000x401
z2 = a1 * Theta1' ;   % (5000*401)*(401*25)
a2 = sigmoid(z2) ;    % 5000*25
a2 = [ones(m,1) a2] ; % 5000*26
z3 = a2 * Theta2' ;   % (5000*26)*(26*10)
a3 = sigmoid(z3);     % 5000*10
h_x= a3 ;             % 5000*10

%[max_val temp4] = max(a3,[],2) ;% temp4-> 5000*1 (indices of max candidate)temp5 = 0 ;
k=1:num_labels; % 1*10
%expected = (temp4==k) ; % 5000*10
yn = (y==k) ;           % 5000*10

%for i=1:m 
 % temp5 = temp5 - ( ( yn(i,:) * log(a3(i,:)') ) + ( ( 1 - yn(i,:) ) * log( 1 - (a3(i,:)') ) ) );
%endfor

 J = (1/m) * sum(sum(- yn .* log(h_x) - (1-yn) .* log(1-h_x) ));


reg1 = sum(sum( (Theta1(:,2:input_layer_size+1)).^2 ) );
reg2 = sum(sum( (Theta2(:,2:hidden_layer_size+1)).^2 ));



J = J + ( (lambda/(2*m))*(reg1+reg2) );



% -------------------------------------------------------------

%yn = (y==k)   where k = 1:num_labels 

for t = 1:m
    a_1= X(t,:)' ; % 400*1
    a_1= [1;a_1]; % 401*1
    
    z_2= Theta1 * a_1 ; % (25*401)*(401*1)
    a_2= sigmoid(z_2); % 25*1
    a_2= [1;a_2];   %  26*1
    
    z_3= Theta2 * a_2;  % (10*26)*(26*1)
    a_3= sigmoid(z_3); % 10*1
    
    y_t= yn(t,:)' ;
       
      
    %temp5 = temp5 - ( ( y_t(i,:) * log(a3(i,:)') ) + ( ( 1 - yn(i,:) ) * log( 1 - (a3(i,:)') ) ) );

    del3= a_3 - y_t ;  % 10*1
    %thet2= Theta2(:,2:end) ;
    del2= (Theta2' * del3) ;  % (26*10)*(10*1)
   % del2= del2(2:end);    % 25*1
    del2= del2 .* sigmoidGradient([1 ; z_2]);
    del2= del2(2:end) ;
    Theta2_grad= Theta2_grad + ( del3 * a_2' ) ;
    Theta1_grad= Theta1_grad + ( del2 * a_1' ) ;
end;


Theta1_grad = (1/m) * Theta1_grad ;
Theta2_grad = (1/m) * Theta2_grad ;




% =========================================================================

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ( (lambda/m) * Theta1(:,2:end) );
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ( (lambda/m) * Theta2(:,2:end) );


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
