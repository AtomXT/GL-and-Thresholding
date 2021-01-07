function [S, A, Sigma_res] = Closed_form(X, tau)

% If we allocate more than this number times n nonzeros, terminate
% prematurely.
MEMORY_LIMIT = 500;

% Get partition size
[n, m] = size(X);
skip = ceil(sqrt(n*MEMORY_LIMIT));

display('phase 1: done')

% Partition x into cells
tmp = floor(n/skip);
rows = [repmat(skip,tmp,1); n - tmp*skip];
cumrows = [0;cumsum(rows)];
X = mat2cell(X, rows);
p = numel(X);

% Threshold one submatrix of M = x*x' at a time.
[ii,jj,kk] = deal(cell(1,p)); 
total_nz = 0;

Xd = [];
for j = 1:p
    [this_ii, this_jj, this_kk] = deal(cell(1,j));
    for i = 1:j
	
        % Form matrix
        Mij = X{i}*X{j}'/m;
        if i == j
            Xd = [Xd; diag(Mij)];
            Mij = triu(Mij,1);
        end
		
        % Do soft-thresholding
        [ii_pos, jj_pos, kk_pos] = find(max(Mij-tau,0));
        [ii_neg, jj_neg, kk_neg] = find(min(Mij+tau,0));
        
        % Record nonzeros
        this_ii{i} = [ii_pos; ii_neg] + cumrows(i);
        this_jj{i} = [jj_pos; jj_neg] + cumrows(j);
        this_kk{i} = [kk_pos; kk_neg];
        
		% Sum nonzeros
        total_nz = total_nz + numel(this_ii{i});
        
		% Check for memory issues
        if total_nz > MEMORY_LIMIT * 10*n
            error('REACHED MEMORY LIMIT. EXITING....');
        end
    end
	
    % Assemble this column
    ii{j} = cat(1, this_ii{:});
    jj{j} = cat(1, this_jj{:});
    kk{j} = cat(1, this_kk{:});
end

% Assemble all columns
ii = cat(1, ii{:});
jj = cat(1, jj{:});
kk = cat(1, kk{:});


% form sparse matrix:
% 	A is our closed-form solution.
%	S is the soft thresholded sample covariance matirx.
% 	Sigma_res is the hard thresholded sample covariance matrix.

S = sparse(ii,jj,kk,n,n);
Sigma_res = sparse(ii,jj,kk+tau*sign(kk),n,n);
[ii_a, jj_a, kk_a] = find(S);
aa = -kk_a./(Xd(ii_a).*Xd(jj_a)-kk_a.^2);
A = sparse(ii_a,jj_a,aa,n,n);
A = A + A';
S = S + S';
Sigma_res = Sigma_res + Sigma_res';
Atemp = transpose(1+sum(-A.*S))./(sparse(Xd));

A = A + diag(sparse(Atemp));
S = S + diag(sparse(Xd));
Sigma_res = Sigma_res + diag(sparse(Xd));
end