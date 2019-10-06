function [freqs] = find_freq_sdp_thresh(y, tol)
addpath(genpath('/usr/local/MATLAB/cvx')) ;

y = y';
N = size(y, 1);

P = 1000;
u = (0:P-1)'/P;

Fourier = @(N,x)exp(-2i*pi*(0:N-1)'*x(:)');

Phi  = @(N,x,a)Fourier(N,x)*a;
PhiS = @(N,u,p)real( Fourier(N,u)'* p );

% Parameters for SDP resolution
q = ones(N,1);
alpha = 2 ;

% SDP CBLasso
lambda_max_CBlasso = norm(PhiS(N,u,y),inf) /(norm(y)*sqrt(N))  ;
lambda_CBlasso = lambda_max_CBlasso/alpha ;
cvx_precision best
cvx_solver sdpt3
cvx_begin sdp quiet
    variable X(N+1,N+1) hermitian;
    variable c(N) complex;
    X >= 0;
    X(N+1,N+1) == 1;
    X(1:N,N+1) == c .* conj(q);
    trace(X) == 2;
    norm(c)<= 1/(sqrt(N)*lambda_CBlasso) ;
    for j = 1:N-1,
        sum(diag(X,j)) == X(N+1-j,N+1);
    end
    maximize(real(c'*y))
cvx_end

CBlasso_coeff = X(1:N,N+1);

% CBlasso poly
aux =- conv(CBlasso_coeff,flipud(conj(CBlasso_coeff))) ; % poly coeff of $- |p|^2$
aux(N)=1+aux(N);
roots_all_CBlasso = roots(flipud(aux)); % roots function convention


% CBlasso: Isolate roots on the unit circle
roots_CBlasso_detected = roots_all_CBlasso(abs(1-abs(roots_all_CBlasso)) < tol);
if length(roots_CBlasso_detected)==0
	[~, idx_ang] = sort(angle(roots_all_CBlasso));
	roots_CBlasso_ang = roots_all_CBlasso(idx_ang);
	roots_CBlasso_ang = roots_CBlasso_ang(1:2:end);
	[~, idx_mag] = sort(abs(1-abs(roots_CBlasso_ang)));
	roots_CBlasso_detected = roots_CBlasso_ang(idx_mag);
	roots_CBlasso_detected = roots_CBlasso_detected(1:1);
end
[~,I]=sort(angle(roots_CBlasso_detected));
roots_CBlasso_detected = roots_CBlasso_detected(I); % roots of 1-|p|^2
roots_CBlasso_detected = roots_CBlasso_detected(1:2:end); % roots of 1-|p

% Precomputation for CBlasso
x_CBlasso = angle(roots_CBlasso_detected)/(2*pi);
x_CBlasso = sort(mod(x_CBlasso+0.5,1)-0.5);

freqs = x_CBlasso;
end
