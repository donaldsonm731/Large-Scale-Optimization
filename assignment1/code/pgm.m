% GM with a fixed step
%
% Least squares: gradient method with fixed step
%
% U. S. Kamilov, CIG, WUSTL, 2021.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% prepare workspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; home;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load the variables of the optimization problem
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('dataset.mat');

[m, n] = size(A); % m rows, n cols

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% set up the function and its gradient (*** edit this ***)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

evaluateFunc = @(x) (1/2)*norm(A*x-b)^2;
evaluateGrad = @(x) A'*A*x - A'*b;
proj_f = @(x) max(x,0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% parameters of the gradient method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xInit = zeros(n, 1); % zero initialization
stepSize = 1/(norm(A,2)^2); % step-size of the gradient method (*** edit this ***)
tol = 1e-4; % stopping tolerance
maxIter = 200; % maximum number of iterations

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% optimize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialize
x = xInit;

% keep track of cost function values
objVals = zeros(maxIter, 1);
infErrs = zeros(maxIter, 1);

% iterate
for iter = 1:maxIter
    
    % gradient at w
    grad = evaluateGrad(x);
    
    % update using GM(*** edit this ***)
    %xNext = x - stepSize*grad;
    
    % update using PGM
    xNext = proj_f(x - stepSize*grad);
    
    % evaluate the objective
    funcNext = evaluateFunc(xNext);
    
    % store the objective and the classification error
    objVals(iter) = funcNext;
    infErrs(iter) = norm(x(:)-xtrue(:))/norm(xtrue(:));
   
    fprintf('[%d/%d] [step: %.1e] [objective: %.1e]\n',...
        iter, maxIter, stepSize, objVals(iter));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % begin visualize data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % plot the evolution
    figure(1);
    set(gcf, 'Color', 'w');
    subplot(2, 2, 1:2);
    stem(1:n, xtrue);
    hold on;
    stem(1:n, x, 'r*');
    hold off;
    xlim([1, n])
    subplot(2, 2, 3);
    semilogy(1:iter, objVals(1:iter), 'b-',...
        iter, objVals(iter), 'b*', 'LineWidth', 2);
    grid on;
    axis tight;
    xlabel('iteration');
    ylabel('objective');
    title(sprintf('cost: %.4e', objVals(iter)));
    xlim([1 maxIter]);
    set(gca, 'FontSize', 16);
    subplot(2, 2, 4);
    semilogy(1:iter, infErrs(1:iter), 'r-',...
        iter, infErrs(iter), 'r*', 'LineWidth', 2);
    grid on;
    axis tight;
    xlabel('iteration');
    ylabel('normalized error');
    title(sprintf('err: %.2e', infErrs(iter)));
    xlim([1 maxIter]);
    set(gca, 'FontSize', 16);
    drawnow;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % end visualize data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % update w
    x = xNext;
end
