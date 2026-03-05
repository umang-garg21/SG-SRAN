%% ============================================================
% Debug Quaternion Convention from .mat (converted from .npy)
% Determines:
%   - component order (xyzw vs wxyz)
%   - whether stored orientation matches MTEX (crystal->specimen) or Bunge (specimen->crystal)
%
% MTEX convention: crystal -> specimen (inverse of Bunge)
% Bunge convention: specimen -> crystal
% They are inverses. Euler angles reported are the usual Bunge ZXZ angles.
%% ============================================================
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

%% ============================================================
% STEP 2 — Build MTEX orientation objects
% ============================================================
cs = crystalSymmetry('m-3m', 'mineral', 'FCC');
sR = cs.fundamentalSector;       % sphericalRegion (red wedge)  :contentReference[oaicite:1]{index=1}
ss = specimenSymmetry('triclinic');

q_vec = reshape(q_wxyz, [], 4);
q_obj = quaternion(q_vec(:,1), q_vec(:,2), q_vec(:,3), q_vec(:,4)); % MTEX expects (w,x,y,z)

ori = orientation(inv(q_obj), cs, ss);    % specimen -> crystal (Bunge-equivalent)

q = q_obj;

%% 4) Choose a sample direction for the IPF (commonly Z)
% IPF asks: "which CRYSTAL direction is parallel to a given SAMPLE direction?"
rS = zvector;                    % sample direction (change to xvector/yvector if desired)

% Crystal direction corresponding to sample direction rS:
% (this is the inverse-pole-figure mapping)
vC = ori \ rS;                    % equivalent to inv(ori) * rS (crystal-frame direction)

% Make it a Miller direction so we can use project2FundamentalRegion like the docs
h = Miller(vC, cs);

%% 5) Check which are inside the fundamental sector + project into it
inside = sR.checkInside(h);       % true if already in the red sector

% Project ALL points into the fundamental sector (returns symmetry op too, if requested)
[hFZ, symUsed] = h.project2FundamentalRegion;

%% 6) Plot: red sector + original points + projected points
figure('Color','w','Name','Quaternions in Fundamental Sector (IPF)');
plot(cs); hold on;                          % optional: shows symmetry reference like docs
plot(sR,'color','red');                     % the red fundamental sector wedge :contentReference[oaicite:4]{index=4}

% Original (before projection): inside vs outside
plot(h( inside),'MarkerFaceColor','y', 'Marker','.', 'MarkerSize',12);         % already inside
plot(h(~inside),'MarkerFaceColor','g', 'Marker','o', 'MarkerSize',5);          % outside

% Projected (after projection): where everything lands in the wedge
plot(hFZ(~inside),'MarkerFaceColor','b', 'Marker','s', 'MarkerSize',5, 'LineWidth',1);

title('IPF points from quaternions: original vs projected into the fundamental sector');
hold off
% legend show
fprintf('Already inside sector: %d / %d\n', nnz(inside), numel(inside));