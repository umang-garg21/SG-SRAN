load("ebsd_data.mat")          % loads variable 'data' (H,W,4)
[H,W,~] = size(data);

% ---- Define symmetries (name the phase!) ----
cs = crystalSymmetry('m-3m', 'mineral', 'FCC', 'color', 'light blue');

% cs = crystalSymmetry('m-3m');
ss = specimenSymmetry('triclinic');

% ---- Read quaternions (ASSUMING x,y,z,w) ----
% w  = data(:,:,4);  xq = data(:,:,1);  yq = data(:,:,2);  zq = data(:,:,3);
w  = data(:,:,4);  xq = data(:,:,1);  yq = data(:,:,2);  zq = data(:,:,3);


%%
% Normalize (important)
nrm = sqrt(w.^2 + xq.^2 + yq.^2 + zq.^2);
w = w./nrm; xq = xq./nrm; yq = yq./nrm; zq = zq./nrm;

% Vectorize -> MTEX rotation
q   = quaternion(w(:), xq(:), yq(:), zq(:));
% rot = rotation(q); % This is if you are not in Bunge convention
rot = inv(rotation(q));

% Positions
[xg, yg] = meshgrid(1:W, 1:H);

yg = H - yg + 1;    % top row -> H, bottom row -> 1 (vertical flip)

pos = vector3d(xg(:), yg(:), zeros(numel(xg),1));

% ---- Single-phase EBSD (no notIndexed) ----
CSList  = {cs};                          % only one phase
phaseId = ones(numel(xg),1);             % all pixels are phase #1
prop    = struct();

ebsd = EBSD(pos, rot, phaseId, CSList, prop);
ebsd = ebsd.gridify;

% Confirm phase name is correct
ebsd

%% Plot ALL m-3m

figure('Name','m-3m NO INV', 'Color', 'w');

% ---- IPF XYZ in one MTEX multiplot figure ----
% mtexFig = newMtexFigure('layout',[1 3]);   % 1 row, 3 cols
mtexFig = newMtexFigure('layout',[1 4], ...
    'spacing',0.2,'padding',0.2);


ipfKey = ipfHSVKey(cs);

dirs   = {xvector, yvector, zvector};
titles = {'IPF X','IPF Y','IPF Z'};

for i = 1:3
    nextAxis;  % MTEX-controlled axis slot
    ipfKey.inversePoleFigureDirection = dirs{i};
    plot(ebsd, ipfKey.orientation2color(ebsd.orientations))
    
    axis image off
    title(titles{i})
end

nextAxis;  % MTEX-controlled axis slot
plot(ipfKey);
title('IPF Z key (m-3m)','FontWeight','bold','FontSize',18);
mtexFig.drawNow;

%% Plot ALL O

cs_432 = crystalSymmetry('432');
ipfKey_432 = ipfHSVKey(cs_432);

% mtexFig = newMtexFigure('layout',[1 3]);   % 1 row, 3 cols
mtexFig = newMtexFigure('layout',[1 3], ...
    'spacing',0.12,'padding',0.14);

dirs   = {xvector, yvector, zvector};
titles = {'IPF X','IPF Y','IPF Z'};

for i = 1:3
    nextAxis;  % MTEX-controlled axis slot
    ipfKey_432.inversePoleFigureDirection = dirs{i};
    plot(ebsd, ipfKey_432.orientation2color(ebsd.orientations))

    axis image off
    title(titles{i})
end
mtexFig.drawNow;
annotation('textbox', ...
    [0.39 0.91 0.22 0.05], ...   % [x y w h] normalized figure units
    'String',ipfKey_432.CS1.pointGroup, ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','bottom', ...
    'EdgeColor','none', ...
    'FontWeight','bold', ...
    'FontSize',14);

%% ---- IPF maps: separate figures ----
ipfKey = ipfHSVKey(cs);

dirs   = {xvector, yvector, zvector};
names  = {'IPF X', 'IPF Y', 'IPF Z'};

for i = 1:3
    figure('Name', ['m-3m ' names{i}], 'Color', 'w');

    ipfKey.inversePoleFigureDirection = dirs{i};

    plot(ebsd, ipfKey.orientation2color(ebsd.orientations));
    axis image off

    title(names{i}, 'Interpreter','none');

    ax = gca;
    ax.Units = 'normalized';
    ax.Position = [0.02 0.02 0.96 0.90];   % <-- more top margin
end

%% ---- IPF Z keys for cubic symmetries ----

% m-3m (Oh)
cs_m3m = crystalSymmetry('m-3m');
ipfKey_m3m = ipfHSVKey(cs_m3m);
ipfKey_m3m.inversePoleFigureDirection = zvector;

figure('Name','IPF Z key: m-3m','Color','w');
plot(ipfKey_m3m);
title('IPF Z key (m-3m)');

% 432 (O)
cs_432 = crystalSymmetry('432');
ipfKey_432 = ipfHSVKey(cs_432);
ipfKey_432.inversePoleFigureDirection = zvector;

figure('Name','IPF Z key: 432','Color','w');
plot(ipfKey_432);
title('IPF Z key (432)');

%% ---- Sanity check ----
% figure
% plotPDF(ebsd.orientations, Miller(1,0,0,cs))
