function report = calc_fcc_seeds_mtex(varargin)
%CALC_FCC_SEEDS_MTEX Compute FCC invariant seeds (L=4, L=6) in MTEX.
%
% Uses Reynolds projection over the 24 proper cubic rotations:
%   P_l = (1/|G|) * sum_g D^l(g)
% and extracts the eigenvector of P_l with eigenvalue near 1.
%
% Name-Value Options
%   'UseNormalizedWigner' : false (default)
%
% Output
%   report : struct with computed seeds and comparison metrics.

opts = parse_opts(varargin{:});

[s4, eval4] = compute_reynolds_seed(get_fcc_symmetry_rotations(), 4, opts.UseNormalizedWigner);
[s6, eval6] = compute_reynolds_seed(get_fcc_symmetry_rotations(), 6, opts.UseNormalizedWigner);

s4_py = zeros(9, 1); s4_py(5) = 0.7638; s4_py(9) = 0.6455; s4_py = s4_py / norm(s4_py);
s6_py = zeros(13, 1); s6_py(7) = 0.3536; s6_py(11) = -0.9354; s6_py = s6_py / norm(s6_py);

s4_exact = zeros(9, 1); s4_exact(5) = sqrt(7 / 12); s4_exact(9) = sqrt(5 / 12); s4_exact = s4_exact / norm(s4_exact);
s6_exact = zeros(13, 1); s6_exact(7) = 1 / sqrt(8); s6_exact(11) = -sqrt(7 / 8); s6_exact = s6_exact / norm(s6_exact);

report = struct();
report.seed_l4 = s4;
report.seed_l6 = s6;
report.max_eig_l4 = eval4;
report.max_eig_l6 = eval6;
report.l2_l4_vs_python = norm(s4 - s4_py);
report.l2_l6_vs_python = norm(s6 - s6_py);
report.maxabs_l4_vs_python = max(abs(s4 - s4_py));
report.maxabs_l6_vs_python = max(abs(s6 - s6_py));
report.l2_l4_vs_exact = norm(s4 - s4_exact);
report.l2_l6_vs_exact = norm(s6 - s6_exact);
report.maxabs_l4_vs_exact = max(abs(s4 - s4_exact));
report.maxabs_l6_vs_exact = max(abs(s6 - s6_exact));

fprintf('\nFCC seed verification (MTEX)\n');
fprintf('============================================================\n');
fprintf('L=4 max eigenvalue: %.16f\n', eval4);
fprintf('  nonzero idx (1-based): %s\n', mat2str(find(abs(s4) > 1e-10).'));
fprintf('  values: %s\n', mat2str(s4(find(abs(s4) > 1e-10)).', 16));
fprintf('  vs python  L2=%.3e  maxabs=%.3e\n', report.l2_l4_vs_python, report.maxabs_l4_vs_python);
fprintf('  vs exact   L2=%.3e  maxabs=%.3e\n', report.l2_l4_vs_exact, report.maxabs_l4_vs_exact);
fprintf('\n');
fprintf('L=6 max eigenvalue: %.16f\n', eval6);
fprintf('  nonzero idx (1-based): %s\n', mat2str(find(abs(s6) > 1e-10).'));
fprintf('  values: %s\n', mat2str(s6(find(abs(s6) > 1e-10)).', 16));
fprintf('  vs python  L2=%.3e  maxabs=%.3e\n', report.l2_l6_vs_python, report.maxabs_l6_vs_python);
fprintf('  vs exact   L2=%.3e  maxabs=%.3e\n', report.l2_l6_vs_exact, report.maxabs_l6_vs_exact);
fprintf('============================================================\n');

end

function opts = parse_opts(varargin)
p = inputParser;
addParameter(p, 'UseNormalizedWigner', false);
parse(p, varargin{:});
opts = p.Results;
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
rot_syms = rotation(quaternion(q_wxyz(:, 1), q_wxyz(:, 2), q_wxyz(:, 3), q_wxyz(:, 4)));
end

function [seed, max_eval] = compute_reynolds_seed(rot_syms, l, use_normalized_wigner)
d = 2 * l + 1;
U = real_to_complex_change_of_basis(l);
P = zeros(d, d);

for k = 1:numel(rot_syms)
    if use_normalized_wigner
        Dc = WignerD(rot_syms(k), l, 'normalize');
    else
        Dc = WignerD(rot_syms(k), l);
    end
    Dr = U' * Dc * U;
    Dr = real(0.5 * (Dr + conj(Dr)));
    P = P + Dr;
end

P = P / numel(rot_syms);
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

function U = real_to_complex_change_of_basis(l)
d = 2 * l + 1;
U = complex(zeros(d, d));
idx = @(m) (m + l + 1);
U(idx(0), idx(0)) = 1.0;
for m = 1:l
    a = (-1)^m;
    U(idx(m), idx(m)) = a / sqrt(2);
    U(idx(m), idx(-m)) = 1i * a / sqrt(2);
    U(idx(-m), idx(m)) = 1 / sqrt(2);
    U(idx(-m), idx(-m)) = -1i / sqrt(2);
end
end
