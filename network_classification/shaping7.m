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

% ==============================
% ==============================
% Visualization
% ==============================
% ==============================

figure;
hold on;
axis equal;

% Unit circle
t = linspace(0, 2*pi, 500);
plot(cos(t), sin(t), '--', ...
    'Color', [0.75 0.75 0.75], ...
    'LineWidth', 1.2);

% Initial points
scatter(1, 0, 130, 'filled');
scatter(-1, 0, 130, 'filled');

% Labels
text(1.10, 0.08, 'A', ...
    'FontSize', 14, 'FontWeight', 'bold');

text(-1.18, 0.08, 'B', ...
    'FontSize', 14, 'FontWeight', 'bold');

% ==============================
% Curved arrows with arrowheads
% ==============================

start_angles = [0, pi];

for i = 1:2

    theta = linspace(start_angles(i) + 0.12, ...
                     start_angles(i) + 0.75, 100);

    x = cos(theta);
    y = sin(theta);

    % Curved black line
    plot(x, y, 'k', 'LineWidth', 2);

    % ----- Arrowhead at the END of the curve -----

    % End point
    tip = [x(end), y(end)];

    % Tangent direction at the end
    direction = tip - [x(end-3), y(end-3)];
    direction = direction / norm(direction);

    % Perpendicular direction
    perpendicular = [-direction(2), direction(1)];

    % Arrowhead size
    headLength = 0.09;
    headWidth  = 0.05;

    % Base of triangle
    base = tip - headLength * direction;

    % Triangle corners
    p1 = tip;
    p2 = base + headWidth * perpendicular;
    p3 = base - headWidth * perpendicular;

    % Draw arrowhead
    patch([p1(1) p2(1) p3(1)], ...
          [p1(2) p2(2) p3(2)], ...
          'k', ...
          'EdgeColor', 'k');
end

% Coordinate axes
plot([-1.3 1.3], [0 0], 'k', 'LineWidth', 0.8);
plot([0 0], [-1.3 1.3], 'k', 'LineWidth', 0.8);

% Axis labels
text(1.32, -0.08, 'x_1', 'FontSize', 13);
text(0.05, 1.28, 'x_2', 'FontSize', 13);

xlim([-1.4 1.4]);
ylim([-1.4 1.4]);

set(gca, 'XTick', [], 'YTick', []);
set(gca, 'XColor', 'none', 'YColor', 'none');

box off;
set(gcf, 'Color', 'w');