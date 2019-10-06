function [etaLambda_CBlasso] = find_freq_sdp(y)
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
tic
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
toc

CBlasso_coeff = X(1:N,N+1);

etaLambda_CBlasso = [PhiS(N,u,CBlasso_coeff), u];
end

