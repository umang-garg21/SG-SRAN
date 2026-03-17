% DEMO_FCC_ENCODER_MTEX
% Minimal usage example for fcc_encoder_mtex.m

% Example quaternions (w,x,y,z), scalar-first.
% q(1,:) = identity rotation.
q = [
    1.0 0.0 0.0 0.0;
    cosd(15) sind(15) 0.0 0.0
];

[f4, f6, meta] = fcc_encoder_mtex(q, ...
    'InputConvention', 'bunge_passive', ...
    'UseNormalizedWigner', false, ...
    'SeedMode', 'computed');

disp('f4:');
disp(f4);

disp('f6:');
disp(f6);

disp('meta:');
disp(meta);

disp('Seed verification (computed vs exact):');
disp(meta.seed_info);
