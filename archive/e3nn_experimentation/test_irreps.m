%% test_irreps_truncation_fcc.m
% Demonstrate irrep-truncation error in the FCC cubic invariant feature space:
%   f_l(q) = D^l(R(q)) * s_l   (real tesseral basis, Reynolds seed s_l)
%
% This script avoids vectorized WignerD(R,l) since your MTEX returns [1 1] for arrays.
% It loops exactly like your calc_fcc_seeds_mtex.

% startup_mtex;  %#ok<*UNRCH>
rng(0);

%% -----------------------
% 1) Settings
%% -----------------------
Ls_true  = [4 6 8 10 12];       % degrees present in the "true" signal
Ls_tests = { [4], [4 6], [4 6 8], [4 6 8 10], [4 6 8 10 12] };
names    = { 'L4', 'L4+L6', 'L4+L6+L8', 'L4+L6+L8+L10', 'L4+...+L12' };

Ntrain = 2500;
Ntest  = 1500;

use_normalized_wigner = false;  % match your code (can set true)

%% -----------------------
% 2) Generate random quaternions (active) and rotations
%% -----------------------
q_train = rand_unit_quats(Ntrain);  % N x 4 wxyz
q_test  = rand_unit_quats(Ntest);

rot_train = rotation(quaternion(q_train(:,1), q_train(:,2), q_train(:,3), q_train(:,4)));
rot_test  = rotation(quaternion(q_test(:,1),  q_test(:,2),  q_test(:,3),  q_test(:,4)));

%% -----------------------
% 3) Compute Reynolds seeds s_l for all needed l
%% -----------------------
rot_syms = get_fcc_symmetry_rotations();

seeds = struct();
for l = Ls_true
    [s, evalmax] = compute_reynolds_seed(rot_syms, l, use_normalized_wigner);
    seeds.(sprintf('s%d', l)) = s;
    fprintf('Seed l=%d: rank-1 invariant dir found, max eigenvalue %.12f\n', l, evalmax);
end

%% -----------------------
% 4) Encode invariant features f_l(q) for train/test
%% -----------------------
Ftrain_true = build_feature_matrix(rot_train, Ls_true, seeds, use_normalized_wigner);
Ftest_true  = build_feature_matrix(rot_test,  Ls_true, seeds, use_normalized_wigner);

% Define a "ground truth" scalar function that depends on multiple irreps
% y = sum_l a_l * (w_l^T f_l) + noise
% This guarantees truncating L will lose predictive power.
a = randn(size(Ftrain_true,2), 1);
a = a / norm(a);

y_train = Ftrain_true * a;
y_test  = Ftest_true  * a;

% Optional noise (set to 0 for clean demo)
sigma = 0.00;
y_train = y_train + sigma * std(y_train) * randn(size(y_train));
y_test  = y_test  + sigma * std(y_test)  * randn(size(y_test));

%% -----------------------
% 5) Fit with truncated degree sets and report error
%% -----------------------
relRMSE = zeros(numel(Ls_tests), 1);
absRMSE = zeros(numel(Ls_tests), 1);
nFeat   = zeros(numel(Ls_tests), 1);

fprintf('\nTruncation study (fit y using selected irreps):\n');
fprintf('============================================================\n');

for k = 1:numel(Ls_tests)
    Lset = Ls_tests{k};

    Xtr = build_feature_matrix(rot_train, Lset, seeds, use_normalized_wigner);
    Xte = build_feature_matrix(rot_test,  Lset, seeds, use_normalized_wigner);

    nFeat(k) = size(Xtr, 2);

    % least squares fit
    w = Xtr \ y_train;
    yhat = Xte * w;

    e = yhat - y_test;
    absRMSE(k) = sqrt(mean(e.^2));
    relRMSE(k) = absRMSE(k) / (sqrt(mean(y_test.^2)) + eps);

    fprintf('%-14s  features=%5d  AbsRMSE=%.4e  RelRMSE=%.4f\n', ...
        names{k}, nFeat(k), absRMSE(k), relRMSE(k));
end

fprintf('============================================================\n\n');

%% -----------------------
% 6) Plot
%% -----------------------
figure('Color','w');
plot(1:numel(Ls_tests), relRMSE, '-o', 'LineWidth', 1.5);
grid on;
xticks(1:numel(Ls_tests));
xticklabels(names);
xtickangle(20);
ylabel('Relative RMSE');
title('Irrep truncation error in FCC invariant feature space');

% Fix your interpreter warning: use tex (default) and single \le if needed
xlabel('Feature set (increasing retained irreps)', 'Interpreter','tex');

%% =======================================================================
% Local functions (adapted from your working calc_fcc_seeds_mtex)
%% =======================================================================

function q = rand_unit_quats(N)
    q = randn(N,4);
    q = q ./ max(vecnorm(q,2,2), eps);
    % Make w>=0 canonical (optional)
    flip = q(:,1) < 0;
    q(flip,:) = -q(flip,:);
end

function X = build_feature_matrix(rot, Lset, seeds, use_normalized_wigner)
    % Concatenate [f_l] for each l in Lset
    X = [];
    for l = Lset
        s = seeds.(sprintf('s%d', l));
        f = encode_degree(rot, l, s, use_normalized_wigner);  % N x (2l+1)
        X = [X, f]; %#ok<AGROW>
    end
    % Normalize columns (helps conditioning)
    cn = sqrt(sum(X.^2, 1)) + eps;
    X = X ./ cn;
end

function f = encode_degree(rot, l, s_real, use_normalized_wigner)
    d = 2*l + 1;
    n = numel(rot);
    f = zeros(n, d);

    U = real_to_complex_change_of_basis(l);

    for i = 1:n
        if use_normalized_wigner
            Dc = WignerD(rot(i), l, 'normalize');
        else
            Dc = WignerD(rot(i), l);
        end

        if ~ismatrix(Dc) || any(size(Dc) ~= [d d])
            error('Unexpected WignerD output for l=%d: got [%s].', l, num2str(size(Dc)));
        end

        % Complex->real tesseral basis
        Dr = U' * Dc * U;

        % Feature vector in real basis, using Reynolds seed
        f(i,:) = real(Dr * s_real).';
    end
end

function rot_syms = get_fcc_symmetry_rotations()
    % 24 proper cubic rotations (your exact list)
    inv_sqrt_2 = 1 / sqrt(2);
    half = 0.5;

    q_wxyz = [
        1, 0, 0, 0;
        0, 1, 0, 0;
        0, 0, 1, 0;
        0, 0, 0, 1;
        inv_sqrt_2,  inv_sqrt_2, 0, 0;
        inv_sqrt_2, 0,  inv_sqrt_2, 0;
        inv_sqrt_2, 0, 0,  inv_sqrt_2;
        inv_sqrt_2, -inv_sqrt_2, 0, 0;
        inv_sqrt_2, 0, -inv_sqrt_2, 0;
        inv_sqrt_2, 0, 0, -inv_sqrt_2;
        0,  inv_sqrt_2,  inv_sqrt_2, 0;
        0,  inv_sqrt_2, 0,  inv_sqrt_2;
        0, 0,  inv_sqrt_2,  inv_sqrt_2;
        0,  inv_sqrt_2, -inv_sqrt_2, 0;
        0, 0,  inv_sqrt_2, -inv_sqrt_2;
        0,  inv_sqrt_2, 0, -inv_sqrt_2;
        half,  half,  half,  half;
        half, -half, -half,  half;
        half, -half,  half, -half;
        half,  half, -half, -half;
        half,  half,  half, -half;
        half,  half, -half,  half;
        half, -half,  half,  half;
        half, -half, -half, -half;
    ];

    q_obj = quaternion(q_wxyz(:,1), q_wxyz(:,2), q_wxyz(:,3), q_wxyz(:,4));
    rot_syms = rotation(q_obj);
end

function [seed, max_eval] = compute_reynolds_seed(rot_syms, l, use_normalized_wigner)
    % Reynolds projector in REAL tesseral basis:
    %   P_l = (1/|G|) sum_{g in G} D^l(g)  (converted to real basis)
    d = 2*l + 1;
    U = real_to_complex_change_of_basis(l);
    P = zeros(d,d);

    for k = 1:numel(rot_syms)
        if use_normalized_wigner
            Dc = WignerD(rot_syms(k), l, 'normalize');
        else
            Dc = WignerD(rot_syms(k), l);
        end
        Dr = U' * Dc * U;                         % real basis (complex numeric)
        Dr = real(0.5 * (Dr + conj(Dr)));         % force real symmetric part
        P  = P + Dr;
    end

    P = P / numel(rot_syms);
    P = 0.5 * (P + P.');                          % symmetrize

    [V,E] = eig(P);
    evals = real(diag(E));
    [max_eval, idx] = max(evals);
    seed = real(V(:,idx));

    % Fix sign convention for stability
    if seed(l+1) < 0
        seed = -seed;
    end
    seed(abs(seed) < 1e-10) = 0;
    seed = seed / norm(seed);
end

function U = real_to_complex_change_of_basis(l)
    % Same as your function: c = U * r, where r is real tesseral (m=-l..l)
    d = 2*l + 1;
    U = complex(zeros(d,d));
    idx = @(m) (m + l + 1);

    U(idx(0), idx(0)) = 1.0;

    for m = 1:l
        a = (-1)^m;

        % Y_m = a/sqrt(2) * (R_m + i R_-m)
        U(idx(m),  idx(m))   = a / sqrt(2);
        U(idx(m),  idx(-m))  = 1i * a / sqrt(2);

        % Y_-m = 1/sqrt(2) * (R_m - i R_-m)
        U(idx(-m), idx(m))   = 1 / sqrt(2);
        U(idx(-m), idx(-m))  = -1i / sqrt(2);
    end
end