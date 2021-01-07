clear
clc


load test

Sparse_mod = sparse(abs(invSigma)>0);
display('number of nonzeros in true inverse covariance');
nonzero = sum(sum(triu(Sparse_mod,1)))

K = 0.435;
lambda = K*sqrt(log(dim)/n);

%% %%%%%%%%%%%%%%%%%%%%%%%% Closed-Form %%%%%%%%%%%%%%%%%%%%%%%%%%

display('finding the closed-form solution...')
tic
[S, A, Sigma_res] = Closed_form(x, lambda);

clear x;
display('finding the closed form solution: done')
Time_closed = toc

thresh_nnz = sum(sum(triu(abs(S),1)>0))


%% Errors

% Closed-Form
Sparse_closed = sparse(abs(A)>1e-5);
Sparse_true = sparse(abs(invSigma)>1e-5);

TPR_closed = sum(sum(Sparse_true.*Sparse_closed))/sum(sum(Sparse_true));
FPR_closed = (sum(sum(Sparse_closed))-sum(sum(Sparse_true.*Sparse_closed)))/(size(Sparse_true,1)*size(Sparse_true,2)-sum(sum(Sparse_true)));

display('TPR and FPR for closed form:')
TPR_closed
FPR_closed

E = A-invSigma;
[~, ~, nnz_E] = find(E);
[~, ~, nnz_invSigma] = find(invSigma);

norm_error = norm(nnz_E)/norm(nnz_invSigma);

display('Normalized error norm for closed form:')
norm_error


