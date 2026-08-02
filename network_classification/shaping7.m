close all; clear all

learning_rate =.1;

%shaping

M=50;
%r=rand(M,1)>0.5;
r=0.25;
rate_change=0.45;
angles=[0:M-1]'*rate_change;
X_shaping=[cos(angles) sin(angles)];


thetas_shaping=-ones(M,2);

%thetas_shaping(1,:) = thetas(end,:);
rs=0.114944784966984;
psis=1.569768424213780;
thetas_shaping(1,:)=rs*[cos(psis) -sin(psis)];


epochs=1;
for j1=1:M-1
    predictions = sigmoid(X_shaping(j1,:) * thetas_shaping(j1,:)');
    %y_shaping=.025+.95*(predictions>0.5);
    y_shaping=(predictions>0.5);
    theta = linear_classification_gradient_descent(X_shaping(j1,:), y_shaping, learning_rate, epochs, thetas_shaping(j1,:));
    thetas_shaping(j1+1,:)=theta;
    %biases(i1+1)=bias;
end

%plot(angle(thetas_shaping(:,2)+1i*thetas_shaping(:,3)))
%initial_prediction=(X_shaping*thetas_shaping(1,:)')>0;
alpha=mod(angle(thetas_shaping(:,1)+1i*thetas_shaping(:,2)),2*pi);
plot(mod(3*pi/4+[0:M-1]'*rate_change,2*pi)*180/pi)
hold on;plot(180/pi*alpha,'r')
xlabel('trial')
ylabel('angel')
legend('examples','weights')
title(['change rate=' num2str(rate_change*180/pi) ' degs/trial']);
% figure
% plot(mod(3*pi/4+[0:M-1]'*rate_change-alpha,2*pi)*180/pi)

figure 
plot(sqrt(sum(thetas_shaping.^2,2)))
ylabel('r')
%figure;plot(thetas_shaping)


function [theta] = linear_classification_gradient_descent(X, y, learning_rate, epochs, theta)

    % Gradient Descent
    for epoch = 1:epochs
        % Compute predictions
        predictions = sigmoid(X * theta');

        % Compute gradients
%        grad_theta = 1 / size(X, 1) * sum((predictions - y)) .* X; %cross entropy
        grad_theta =  sum((predictions - y)) .* X +theta; %cross entropy + regularization

        %grad_theta = predictions*(1-predictions) * sum((predictions - y))  %.* X; %least square
        
        
        % Update weights 
        theta = theta - learning_rate * grad_theta;
    end
end

function sigmoid_value = sigmoid(z)
    % Sigmoid function
    sigmoid_value = 1 ./ (1 + exp(-z));
end