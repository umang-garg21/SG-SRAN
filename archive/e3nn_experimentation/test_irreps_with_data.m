%% choose_L_for_cubic_in718.m
% Decide how many irreps L to use for cubic symmetry invariant encoding of quaternions
% (IN718 is cubic m-3m).
%
% Uses the SAME encoding style you are using:
%   f_l(q) = D^l(R(q)) * s_l
% where s_l is a Reynolds-projected invariant seed in the REAL tesseral basis.
%
% Output: table of collision rates vs L-set, and a recommended smallest L-set.

% startup_mtex; %#ok<*UNRCH>
rng(0);

%% =========================
% USER INPUT SECTION



%% =========================
clear; clc; close all;

%% ---- Basic stats ----
load("Open_718_Test_hr_x_block_0.mat")   % expects variable 'data' (H,W,4)
[H,W,C] = size(data);
assert(C==4, 'Expected 4 quaternion components');
fprintf('Shape: %d x %d x %d\n\n', H, W, C);

flat = reshape(data, [], 4);
for i = 1:4
    fprintf('Component %d: mean = %+0.4f  std = %0.4f  min = %+0.4f  max = %+0.4f\n',...
        i, mean(flat(:,i)), std(flat(:,i)), min(flat(:,i)), max(flat(:,i)));
end
fprintf('\n');

%% ============================================================
% STEP 1 — Guess component order
% ============================================================
means = abs(mean(flat));
[~, scalar_idx] = max(means);
fprintf('Likely scalar component index: %d\n\n', scalar_idx);

if scalar_idx == 4
    fprintf('-> Most likely stored as [x y z w] (scalar last)\n\n');
    q_wxyz = cat(3, data(:,:,4), data(:,:,1), data(:,:,2), data(:,:,3)); % -> (w,x,y,z)
elseif scalar_idx == 1
    fprintf('-> Most likely stored as [w x y z] (scalar first)\n\n');
    q_wxyz = data;
else
    warning('Scalar not clearly at 1 or 4 — manual inspection required');
    q_wxyz = data;
end

% Normalize
norms = sqrt(sum(q_wxyz.^2,3));
q_wxyz = q_wxyz ./ norms;

%% Option A: If you already have quats in workspace:
%   q_wxyz = ...   % (N x 4) double, scalar-first [w x y z]
%
% Option B: load from a .mat:
% load('my_in718_quats.mat','q_wxyz');

if ~exist('q_wxyz','var')
    % Fallback demo data (random quaternions). Replace with IN718 quats.
    warning('q_wxyz not found. Using random demo quats. Load/define q_wxyz for IN718.');
    Ndemo = 8000;
    q_wxyz = rand_unit_quats(Ndemo);
end

q_wxyz = double(q_wxyz);
assert(size(q_wxyz,2)==4, 'q_wxyz must be N x 4 in [w x y z] order.');

% Convention: your pipeline often uses Bunge passive -> active via conjugation.
InputConvention = 'bunge_passive';      % 'active' OR 'bunge_passive'
UseNormalizedWigner = false;     % match your earlier code
MaxN = 20000;                    % cap #quats used for evaluation (speed)
QueryCount = 600;                % number of query points for NN-collision test

% Candidate degree sets (even l for cubic)
Lsets = { [4], [4 6], [4 6 8], [4 6 8 10], [4 6 8 10 12] };

% Collision definition thresholds
eps_feat = 5e-3;     % "nearly identical features" threshold (after normalization)
theta0  = 5.0;       % degrees; consider "far apart" if misorientation > theta0

% Recommendation criterion
max_allowed_collision_rate = 0.01;   % 1% collisions
max_allowed_p90_mis_small  = 2.0;    % optional: among near-features, p90 misorientation <= 2°

%% =========================
% PREP: subsample quats
%% =========================
N = size(q_wxyz,1);
if N > MaxN
    idx = randperm(N, MaxN);
    q_wxyz = q_wxyz(idx,:);
    N = MaxN;
end

% Normalize and convert to active if needed
q = q_wxyz ./ max(vecnorm(q_wxyz,2,2), eps);

if strcmpi(InputConvention,'bunge_passive')
    q(:,2:4) = -q(:,2:4); % conjugate for passive->active
elseif ~strcmpi(InputConvention,'active')
    error('InputConvention must be ''active'' or ''bunge_passive''.');
end

% Build MTEX rotations
rot = rotation(quaternion(q(:,1), q(:,2), q(:,3), q(:,4)));

%% =========================
% Cubic symmetry operations (proper 24) and Reynolds seeds
%% =========================
rot_syms = get_fcc_symmetry_rotations(); % 24 proper cubic rotations

% Precompute seeds for all degrees needed
allL = unique([Lsets{:}]);
seeds = struct();
for l = allL
    [s, evalmax] = compute_reynolds_seed(rot_syms, l, UseNormalizedWigner);
    seeds.(sprintf('s%d',l)) = s;
    fprintf('Seed l=%d: max eigenvalue %.12f\n', l, evalmax);
end

% Optional: verify invariance of seeds and features (uncomment if desired)
% for l = allL
%     verify_seed_invariance(rot_syms, l, seeds.(sprintf('s%d',l)), UseNormalizedWigner);
% end

%% =========================
% Evaluate candidate L-sets: collision rates + stats
%% =========================
Q = min(QueryCount, N);
q_idx = randperm(N, Q);

results = cell(numel(Lsets),1);

fprintf('\nCollision study settings:\n');
fprintf('  N=%d quats, Q=%d queries\n', N, Q);
fprintf('  eps_feat=%.2e, theta0=%.2f deg\n\n', eps_feat, theta0);

for k = 1:numel(Lsets)
    Lset = Lsets{k};

    % Encode invariant features for this Lset
    Z = encode_feature_set(rot, Lset, seeds, UseNormalizedWigner);  % N x D
    % normalize rows (important)
    Z = Z ./ max(vecnorm(Z,2,2), eps);

    % Compute collisions via nearest neighbor in feature space
    [collRate, nnFeatDist, nnMisorDeg] = collision_rate_nn(q, Z, rot_syms, q_idx, eps_feat, theta0);

    % Additional stats: among near-features, what are misorientation quantiles?
    maskNear = nnFeatDist < eps_feat;
    if any(maskNear)
        p50 = prctile(nnMisorDeg(maskNear), 50);
        p90 = prctile(nnMisorDeg(maskNear), 90);
        p99 = prctile(nnMisorDeg(maskNear), 99);
    else
        p50 = NaN; p90 = NaN; p99 = NaN;
    end

    results{k} = struct( ...
        'Lset', Lset, ...
        'featDim', size(Z,2), ...
        'collisionRate', collRate, ...
        'p50_mis_near', p50, ...
        'p90_mis_near', p90, ...
        'p99_mis_near', p99);

    fprintf('Lset=%-15s  dim=%4d  collRate=%.4f  mis|near p90=%.2f deg\n', ...
        mat2str(Lset), size(Z,2), collRate, p90);
end

%% =========================
% Recommend smallest Lset satisfying criteria
%% =========================
best = [];
for k = 1:numel(results)
    r = results{k};
    ok1 = (r.collisionRate <= max_allowed_collision_rate);
    ok2 = (isnan(r.p90_mis_near) || r.p90_mis_near <= max_allowed_p90_mis_small);
    if ok1 && ok2
        best = r;
        break;
    end
end

fprintf('\n============================================================\n');
fprintf('Summary table:\n');
fprintf('%-18s %-6s %-12s %-14s\n', 'L-set', 'dim', 'collRate', 'p90(mis|near)');
for k = 1:numel(results)
    r = results{k};
    fprintf('%-18s %-6d %-12.4f %-14.2f\n', ...
        mat2str(r.Lset), r.featDim, r.collisionRate, r.p90_mis_near);
end

fprintf('------------------------------------------------------------\n');
if isempty(best)
    fprintf('No L-set met the criteria. Consider increasing max L (add 14,16) or relax thresholds.\n');
else
    fprintf('RECOMMENDED smallest L-set: %s\n', mat2str(best.Lset));
    fprintf('  dim=%d, collisionRate=%.4f, p90(mis|near)=%.2f deg\n', ...
        best.featDim, best.collisionRate, best.p90_mis_near);
end
fprintf('============================================================\n');

%% =========================
% Optional: plot collisionRate vs Lset index
%% =========================
colls = cellfun(@(r) r.collisionRate, results);
figure('Color','w');
plot(1:numel(Lsets), colls, '-o', 'LineWidth', 1.5);
grid on;
xticks(1:numel(Lsets));
xticklabels(cellfun(@(L) mat2str(L), Lsets, 'UniformOutput', false));
xtickangle(20);
ylabel('Collision rate');
xlabel('L-set');
title('Choosing L for cubic-invariant quaternion encoding (IN718)');

%% =======================================================================
% Local functions (MTEX-compatible; loop-based WignerD like your encoder)
%% =======================================================================

function q = rand_unit_quats(N)
    q = randn(N,4);
    q = q ./ max(vecnorm(q,2,2), eps);
    flip = q(:,1) < 0;
    q(flip,:) = -q(flip,:);
end

function Z = encode_feature_set(rot, Lset, seeds, use_normalized_wigner)
    % Concatenate invariant feature blocks f_l (N x (2l+1)) for each l in Lset.
    Z = [];
    for l = Lset
        s = seeds.(sprintf('s%d',l));
        f = encode_degree(rot, l, s, use_normalized_wigner);
        Z = [Z, f]; %#ok<AGROW>
    end
end

function [collRate, nnFeatDist, nnMisorDeg] = collision_rate_nn(q_wxyz, Z, rot_syms, q_idx, eps_feat, theta0)
    % Nearest-neighbor collision rate:
    % For each i in q_idx, find j minimizing ||Z(i)-Z(j)||, then compute misorientation mod cubic.
    N = size(Z,1);
    Q = numel(q_idx);

    nnFeatDist = zeros(Q,1);
    nnMisorDeg = zeros(Q,1);

    for t = 1:Q
        i = q_idx(t);
        vi = Z(i,:);

        dd = vecnorm(Z - vi, 2, 2);
        dd(i) = inf;
        [mind, j] = min(dd);

        nnFeatDist(t) = mind;
        nnMisorDeg(t) = misorientation_deg_mod_cubic(q_wxyz(i,:), q_wxyz(j,:), rot_syms);
    end

    coll = (nnFeatDist < eps_feat) & (nnMisorDeg > theta0);
    collRate = mean(coll);
end

function angDeg = misorientation_deg_mod_cubic(q1, q2, rot_syms)
    % theta(q1,q2) = min_{s in G} angle(q1, q2 ⊗ s)
    % Use quaternion dot-product formula: theta = 2 acos(|<q1,q2>|)
    % Here we apply right multiplication by symmetry quats.
    q1 = q1 ./ max(norm(q1), eps);
    q2 = q2 ./ max(norm(q2), eps);

    % Convert symmetry rotations to quats
    qs = rot_syms.quaternion; % MTEX quaternion array
    % qs is quaternion class; pull numeric wxyz
    w = qs.a; x = qs.b; y = qs.c; z = qs.d;
    S = [w(:) x(:) y(:) z(:)];

    best = inf;
    for k = 1:size(S,1)
        q2s = quat_mul(q2, S(k,:));
        ang = quat_angle_deg(q1, q2s);
        if ang < best
            best = ang;
        end
    end
    angDeg = best;
end

function q = quat_mul(a, b)
    % Hamilton product of two quats in wxyz, row vectors 1x4.
    aw=a(1); ax=a(2); ay=a(3); az=a(4);
    bw=b(1); bx=b(2); by=b(3); bz=b(4);
    q = [ ...
        aw*bw - ax*bx - ay*by - az*bz, ...
        aw*bx + ax*bw + ay*bz - az*by, ...
        aw*by - ax*bz + ay*bw + az*bx, ...
        aw*bz + ax*by - ay*bx + az*bw ];
end

function ang = quat_angle_deg(q1, q2)
    % theta = 2 acos(|dot(q1,q2)|)
    dotp = abs(sum(q1 .* q2));
    dotp = min(max(dotp, -1.0), 1.0);
    ang = 2 * acos(dotp) * (180/pi);
end

function [f, max_imag] = encode_degree(rot, l, s_real, use_normalized_wigner)
    d = 2*l + 1;
    n = numel(rot);
    f = zeros(n, d);

    U = real_to_complex_change_of_basis(l);
    max_imag = 0.0;

    for i = 1:n
        if use_normalized_wigner
            Dc = WignerD(rot(i), l, 'normalize');
        else
            Dc = WignerD(rot(i), l);
        end
        if ~ismatrix(Dc) || any(size(Dc) ~= [d d])
            error('Unexpected WignerD output shape for l=%d: got [%s].', ...
                l, num2str(size(Dc)));
        end

        Dr = U' * Dc * U;
        max_imag = max(max_imag, max(abs(imag(Dr(:)))));

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
    d = 2*l + 1;
    U = real_to_complex_change_of_basis(l);
    P = zeros(d,d);

    for k = 1:numel(rot_syms)
        if use_normalized_wigner
            Dc = WignerD(rot_syms(k), l, 'normalize');
        else
            Dc = WignerD(rot_syms(k), l);
        end
        Dr = U' * Dc * U;
        Dr = real(0.5 * (Dr + conj(Dr)));
        P  = P + Dr;
    end

    P = P / numel(rot_syms);
    P = 0.5 * (P + P.');   % symmetrize

    [V,E] = eig(P);
    evals = real(diag(E));
    [max_eval, idx] = max(evals);
    seed = real(V(:,idx));

    if seed(l+1) < 0, seed = -seed; end
    seed(abs(seed) < 1e-10) = 0;
    seed = seed / norm(seed);
end

function verify_seed_invariance(rot_syms, l, s, use_normalized_wigner)
    d = 2*l + 1;
    U = real_to_complex_change_of_basis(l);
    errs = zeros(numel(rot_syms),1);

    for k = 1:numel(rot_syms)
        if use_normalized_wigner
            Dc = WignerD(rot_syms(k), l, 'normalize');
        else
            Dc = WignerD(rot_syms(k), l);
        end
        Dr = U' * Dc * U;
        Dr = real(0.5*(Dr + conj(Dr)));
        errs(k) = norm(Dr*s - s);
    end

    fprintf('Seed invariance l=%d: max ||D(g)s-s||=%.3e\n', l, max(errs));
end

function U = real_to_complex_change_of_basis(l)
    d = 2*l + 1;
    U = complex(zeros(d,d));
    idx = @(m) (m + l + 1);
    U(idx(0), idx(0)) = 1.0;

    for m = 1:l
        a = (-1)^m;
        U(idx(m),  idx(m))   = a / sqrt(2);
        U(idx(m),  idx(-m))  = 1i * a / sqrt(2);
        U(idx(-m), idx(m))   = 1 / sqrt(2);
        U(idx(-m), idx(-m))  = -1i / sqrt(2);
    end
end