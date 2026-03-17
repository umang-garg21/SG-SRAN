function [f4, f6, meta] = fcc_encoder_mtex(q_wxyz, varargin)
%FCC_ENCODER_MTEX MTEX implementation of the FCC L=4/L=6 encoder logic.
%
% This mirrors the Python encoder in models/autoencoder.py:
%   1) quaternion -> active rotation
%   2) D^4(R), D^6(R)
%   3) f4 = D^4 * s4, f6 = D^6 * s6
%
% Input
%   q_wxyz : (N x 4) scalar-first quaternions [w x y z]
%
% Name-Value Options
%   'InputConvention'     : 'bunge_passive' (default) or 'active'
%   'UseNormalizedWigner' : false (default). If true, calls WignerD(...,'normalize')
%   'SeedMode'            : 'computed' (default) or 'hardcoded'
%   'VerifyHardcoded'     : true (default). If SeedMode='hardcoded', also compute
%                           Reynolds seeds and report mismatch in meta.
%
% Output
%   f4   : (N x 9)  real L=4 feature vectors in real harmonic basis
%   f6   : (N x 13) real L=6 feature vectors in real harmonic basis
%   meta : struct with diagnostics
%
% Notes
% - MTEX WignerD uses complex harmonics. This function converts to a real
%   tesseral basis before applying real seed vectors.
% - Python code uses passive(Bunge) -> active by quaternion conjugation;
%   that same conversion is applied here by default.
% - By default seeds are computed from cubic symmetry via Reynolds projection
%   (not hard-coded), then used by the encoder.

opts = parse_opts(varargin{:});

validateattributes(q_wxyz, {'numeric'}, {'2d', 'ncols', 4, 'real', 'finite'}, mfilename, 'q_wxyz');
q_wxyz = double(q_wxyz);

% Normalize unit quaternions.
qn = q_wxyz ./ max(vecnorm(q_wxyz, 2, 2), eps);

% Match Python wrapper behavior: Bunge passive -> active via conjugation.
if strcmpi(opts.InputConvention, 'bunge_passive')
    q_act = qn;
    q_act(:, 2:4) = -q_act(:, 2:4);
elseif strcmpi(opts.InputConvention, 'active')
    q_act = qn;
else
    error('InputConvention must be ''bunge_passive'' or ''active''.');
end

if strcmpi(opts.SeedMode, 'computed')
    [s4, s6, seed_meta] = compute_fcc_seeds_reynolds(opts.UseNormalizedWigner);
elseif strcmpi(opts.SeedMode, 'hardcoded')
    [s4, s6] = get_hardcoded_seeds();
    seed_meta = struct();
    if logical(opts.VerifyHardcoded)
        [s4_calc, s6_calc, calc_meta] = compute_fcc_seeds_reynolds(opts.UseNormalizedWigner);
        seed_meta = calc_meta;
        seed_meta.hardcoded_vs_computed_l2_l4 = norm(s4 - s4_calc);
        seed_meta.hardcoded_vs_computed_l2_l6 = norm(s6 - s6_calc);
        seed_meta.hardcoded_vs_computed_maxabs_l4 = max(abs(s4 - s4_calc));
        seed_meta.hardcoded_vs_computed_maxabs_l6 = max(abs(s6 - s6_calc));
    end
else
    error('SeedMode must be ''computed'' or ''hardcoded''.');
end

% Build MTEX rotation array from scalar-first quaternions.
q_obj = quaternion(q_act(:, 1), q_act(:, 2), q_act(:, 3), q_act(:, 4));
rot = rotation(q_obj);

[f4, imag4] = encode_degree(rot, 4, s4, opts.UseNormalizedWigner);
[f6, imag6] = encode_degree(rot, 6, s6, opts.UseNormalizedWigner);

meta = struct();
meta.num_quaternions = size(q_wxyz, 1);
meta.input_convention = opts.InputConvention;
meta.use_normalized_wigner = logical(opts.UseNormalizedWigner);
meta.seed_mode = opts.SeedMode;
meta.max_imag_after_real_basis_l4 = imag4;
meta.max_imag_after_real_basis_l6 = imag6;
meta.seed_l4 = s4;
meta.seed_l6 = s6;
meta.seed_info = seed_meta;

end

function [f, max_imag] = encode_degree(rot, l, s_real, use_normalized_wigner)
d = 2 * l + 1;
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

    if ~ismatrix(Dc) || any(size(Dc) ~= [d, d])
        error('Unexpected WignerD output shape for l=%d: got [%s].', ...
            l, num2str(size(Dc)));
    end

    % Convert complex Wigner D to real harmonic basis:
    % c = U * r  =>  r' = U^H * Dc * U * r
    Dr = U' * Dc * U;
    max_imag = max(max_imag, max(abs(imag(Dr(:)))));

    f(i, :) = real(Dr * s_real).';
end

end

function U = real_to_complex_change_of_basis(l)
% Return unitary U such that c = U * r, where:
%   r: real tesseral coefficients ordered by m = -l..l
%   c: complex coefficients ordered by m = -l..l
%
% Convention used:
%   R_m   = ( (-1)^m Y_m + Y_-m ) / sqrt(2),   m > 0
%   R_-m  = ( (-1)^m Y_m - Y_-m ) / (i*sqrt(2)), m > 0
%   R_0   = Y_0

d = 2 * l + 1;
U = complex(zeros(d, d));

m_vals = -l:l;
idx = @(m) (m + l + 1);

U(idx(0), idx(0)) = 1.0;

for m = 1:l
    a = (-1)^m;

    % Y_m = a/sqrt(2) * (R_m + i R_-m)
    U(idx(m), idx(m))   = a / sqrt(2);
    U(idx(m), idx(-m))  = 1i * a / sqrt(2);

    % Y_-m = 1/sqrt(2) * (R_m - i R_-m)
    U(idx(-m), idx(m))  = 1 / sqrt(2);
    U(idx(-m), idx(-m)) = -1i / sqrt(2);
end

% Basic sanity check (unitary up to numerical tolerance)
if norm(U' * U - eye(d), 'fro') > 1e-10
    error('Internal basis transform U is not unitary for l=%d.', l);
end

% Keep m_vals referenced to make ordering explicit for readers.
% (No-op use to avoid accidental edits to ordering assumptions.)
if numel(m_vals) ~= d
    error('Internal ordering failure for l=%d.', l);
end

end

function opts = parse_opts(varargin)
p = inputParser;
p.FunctionName = mfilename;

addParameter(p, 'InputConvention', 'bunge_passive');
addParameter(p, 'UseNormalizedWigner', false);
addParameter(p, 'SeedMode', 'computed');
addParameter(p, 'VerifyHardcoded', true);

parse(p, varargin{:});
opts = p.Results;

end

function [s4, s6] = get_hardcoded_seeds()
s4 = zeros(9, 1);
s4(5) = 0.7638;
s4(9) = 0.6455;
s4 = s4 / norm(s4);

s6 = zeros(13, 1);
s6(7) = 0.3536;
s6(11) = -0.9354;
s6 = s6 / norm(s6);
end

function [s4, s6, meta] = compute_fcc_seeds_reynolds(use_normalized_wigner)
rot_syms = get_fcc_symmetry_rotations();

[s4, eval4] = compute_reynolds_seed(rot_syms, 4, use_normalized_wigner);
[s6, eval6] = compute_reynolds_seed(rot_syms, 6, use_normalized_wigner);

s4_exact = zeros(9, 1);
s4_exact(5) = sqrt(7 / 12);
s4_exact(9) = sqrt(5 / 12);
s4_exact = s4_exact / norm(s4_exact);

s6_exact = zeros(13, 1);
s6_exact(7) = 1 / sqrt(8);
s6_exact(11) = -sqrt(7 / 8);
s6_exact = s6_exact / norm(s6_exact);

meta = struct();
meta.reynolds_max_eig_l4 = eval4;
meta.reynolds_max_eig_l6 = eval6;
meta.computed_vs_exact_l2_l4 = norm(s4 - s4_exact);
meta.computed_vs_exact_l2_l6 = norm(s6 - s6_exact);
meta.computed_vs_exact_maxabs_l4 = max(abs(s4 - s4_exact));
meta.computed_vs_exact_maxabs_l6 = max(abs(s6 - s6_exact));
meta.seed_l4_exact = s4_exact;
meta.seed_l6_exact = s6_exact;
end

function rot_syms = get_fcc_symmetry_rotations()
inv_sqrt_2 = 1 / sqrt(2);
half = 0.5;

q_wxyz = [
    1, 0, 0, 0;
    0, 1, 0, 0;
    0, 0, 1, 0;
    0, 0, 0, 1;
    inv_sqrt_2, inv_sqrt_2, 0, 0;
    inv_sqrt_2, 0, inv_sqrt_2, 0;
    inv_sqrt_2, 0, 0, inv_sqrt_2;
    inv_sqrt_2, -inv_sqrt_2, 0, 0;
    inv_sqrt_2, 0, -inv_sqrt_2, 0;
    inv_sqrt_2, 0, 0, -inv_sqrt_2;
    0, inv_sqrt_2, inv_sqrt_2, 0;
    0, inv_sqrt_2, 0, inv_sqrt_2;
    0, 0, inv_sqrt_2, inv_sqrt_2;
    0, inv_sqrt_2, -inv_sqrt_2, 0;
    0, 0, inv_sqrt_2, -inv_sqrt_2;
    0, inv_sqrt_2, 0, -inv_sqrt_2;
    half, half, half, half;
    half, -half, -half, half;
    half, -half, half, -half;
    half, half, -half, -half;
    half, half, half, -half;
    half, half, -half, half;
    half, -half, half, half;
    half, -half, -half, -half;
];

q_obj = quaternion(q_wxyz(:, 1), q_wxyz(:, 2), q_wxyz(:, 3), q_wxyz(:, 4));
rot_syms = rotation(q_obj);
end

function [seed, max_eval] = compute_reynolds_seed(rot_syms, l, use_normalized_wigner)
d = 2 * l + 1;
n_ops = numel(rot_syms);
U = real_to_complex_change_of_basis(l);
P = zeros(d, d);

for k = 1:n_ops
    if use_normalized_wigner
        Dc = WignerD(rot_syms(k), l, 'normalize');
    else
        Dc = WignerD(rot_syms(k), l);
    end
    Dr = U' * Dc * U;
    Dr = real(0.5 * (Dr + conj(Dr)));
    P = P + Dr;
end

P = P / n_ops;
P = 0.5 * (P + P.');

[V, E] = eig(P);
evals = real(diag(E));
[max_eval, idx] = max(evals);
seed = real(V(:, idx));

if seed(l + 1) < 0
    seed = -seed;
end

seed(abs(seed) < 1e-10) = 0;
seed = seed / norm(seed);
end
