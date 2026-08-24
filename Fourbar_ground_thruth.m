% MATLAB fourbar mechanism simulation (WITH TCP/IP, WITH DATA DISPLAY & PLOTS)
% Modified: send only theta and theta_dot to C++; save theta_ddot to CSV
clear; clc; close all;

%% Parameters
% gravitational acceleration, Section 2.3
g  = 9.81;

% link lengths L1, L2, L3, L4, Section 2.2
L1 = 2.0;  L2 = 8.0;  L3 = 5.0;  L4 = 10.2;

% link masses m1, m2, m3 for links 1, 2, 3 respectively
mA = 2.0;  mB = 8.0;  mC = 5.0;

% mass moment of inertia about centre of mass for each link (Eq.25)
rodInertia = @(m, L) m * L^2 / 12;

params.g  = g;
params.L1 = L1;  params.L2 = L2;  params.L3 = L3;  params.L4 = L4;
params.mA = mA;  params.mB = mB;  params.mC = mC;

% inertia scalars I1, I2, I3 (Eq.25)
params.I1 = rodInertia(mA, L1);
params.I2 = rodInertia(mB, L2);
params.I3 = rodInertia(mC, L3);

% applied torque amplitude T0 (set to zero, no external torque applied)
params.T0  = 0.0;

% input torque angular frequency (unused since T0 = 0)
params.omega_input_torque = 2 * pi * 0.02;

%% Noise parameters
% standard deviations of position, velocity, and acceleration measurements
% (Eq.40)
rng(42);
sigma_theta     = 1e-3;    % position noise (rad)
sigma_theta_dot = 1e-2;    % velocity noise (rad/s)
sigma_theta_ddot= 0.05;    % acceleration noise (rad/s^2)

%% Time
dt   = 0.001;
tend = 10.0;
tsol = 0:dt:tend;
N    = numel(tsol);

%% TCP/IP CONFIGURATION (Section 4.5)
serverIP   = "169.254.131.136";  % C++ server IP
serverPort = 5556;               % server port

%% CONNECT TO C++ SERVER (Section 4.3)
try
    tcpipClient = tcpip(serverIP, serverPort, 'NetworkRole', 'client');
    tcpipClient.InputBufferSize  = 8192;
    tcpipClient.OutputBufferSize = 8192;
    tcpipClient.Timeout          = 30;
    fopen(tcpipClient);
    fprintf('Connected to C++ Augmented Lagrange server!\n\n');
catch ME
    fprintf('Connection failed: %s\n', ME.message);
    return;
end

%% Initial configuration (closure)
% theta_0 is the initial orientation of link 1 (input crank),
% generalized coordinate z(3) in Eq.15
theta_0 = pi / 3;

% compute initial angles theta2_0, theta3_0 and joint positions from
% kinematic loop closure (Eq.16-23)
[B0, C0, theta2_0, theta3_0] = closure_from_theta1(theta_0, params);

% initial centre-of-mass positions for each link (Eq.15)
Rx1_0 = (L1 / 2) * cos(theta_0);
Ry1_0 = (L1 / 2) * sin(theta_0);
Rx2_0 = 0.5 * (B0(1) + C0(1));
Ry2_0 = 0.5 * (B0(2) + C0(2));
Rx3_0 = L4 + (L3 / 2) * cos(theta3_0);
Ry3_0 =      (L3 / 2) * sin(theta3_0);

% generalized coordinate vector z (9 x 1) (Eq.15)
z_init = [Rx1_0; Ry1_0; theta_0; Rx2_0; Ry2_0; theta2_0; Rx3_0; Ry3_0; theta3_0];

% initial generalized velocity vector zdot (9 x 1)
zdot_init = zeros(9, 1);

% full state vector x0 = [z; zdot] (Section 4.1, Eq.52)
x0   = [z_init; zdot_init];
xsol = zeros(N, 18);
xsol(1, :) = x0';

%% -----------------------------------------------------------------------
%% CYCLE COUNTING SETUP
% -----------------------------------------------------------------------
theta_unwrapped_prev = theta_0;
theta_cumulative     = 0.0;
cycle_count_log      = zeros(N, 1);
frac_cycle_log       = zeros(N, 1);
cycle_count_log(1)   = 0;
frac_cycle_log(1)    = 0;

%% Logs for plotting
theta_true      = zeros(N, 1);
theta_dot_true  = zeros(N, 1);
theta_ddot_true = zeros(N, 1);
theta_noisy     = zeros(N, 1);
theta_dot_noisy = zeros(N, 1);
theta_ddot_noisy= zeros(N, 1);

theta_true(1)     = x0(3);
theta_dot_true(1) = x0(12);

% compute initial acceleration from equations of motion (Eq.14b)
dx0 = fourbarode_lagrange(tsol(1), x0, params);
theta_ddot_true(1) = dx0(12);

% apply measurement noise (Eq.40)
theta_noisy(1)      = theta_true(1)      + sigma_theta      * randn();
theta_dot_noisy(1)  = theta_dot_true(1)  + sigma_theta_dot  * randn();
theta_ddot_noisy(1) = theta_ddot_true(1) + sigma_theta_ddot * randn();

%% Main loop (RK4 integration + TCP transmission + display)
for k = 1:N-1
    t = tsol(k);
    x = xsol(k, :)';

    % RK4 integration of the augmented Lagrangian equations of motion (Eq.52)
    k1 = fourbarode_lagrange(t,          x,              params);
    k2 = fourbarode_lagrange(t + dt/2.0, x + dt/2.0*k1, params);
    k3 = fourbarode_lagrange(t + dt/2.0, x + dt/2.0*k2, params);
    k4 = fourbarode_lagrange(t + dt,     x + dt*k3,      params);
    x_next = x + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4);
    xsol(k+1, :) = x_next';

    % extract theta and theta_dot of link 1 (Eq.15)
    theta_k     = x_next(3);
    theta_dot_k = x_next(12);

    % compute theta_ddot from equations of motion (Eq.14b)
    dxdt_k      = fourbarode_lagrange(t + dt, x_next, params);
    theta_ddot_k= dxdt_k(12);

    % add measurement noise (Eq.40)
    theta_meas      = theta_k      + sigma_theta      * randn();
    theta_dot_meas  = theta_dot_k  + sigma_theta_dot  * randn();
    theta_ddot_meas = theta_ddot_k + sigma_theta_ddot * randn();

    % log true and noisy states
    theta_true(k+1)      = theta_k;
    theta_dot_true(k+1)  = theta_dot_k;
    theta_ddot_true(k+1) = theta_ddot_k;
    theta_noisy(k+1)     = theta_meas;
    theta_dot_noisy(k+1) = theta_dot_meas;
    theta_ddot_noisy(k+1)= theta_ddot_meas;

    % -------------------------------------------------------------------
    % CYCLE COUNTING - update at each time step
    % -------------------------------------------------------------------
    delta_raw       = theta_k - theta_unwrapped_prev;
    delta_unwrapped = mod(delta_raw + pi, 2*pi) - pi;

    theta_cumulative     = theta_cumulative + delta_unwrapped;
    theta_unwrapped_prev = theta_unwrapped_prev + delta_unwrapped;

    cycle_count_log(k+1) = floor(abs(theta_cumulative) / (2*pi));
    frac_cycle_log(k+1)  = mod(abs(theta_cumulative), 2*pi) / (2*pi);
    % -------------------------------------------------------------------

    % display noisy measurements and cycle count
    fprintf('t = %.3f s | theta = %.6f rad | theta_dot = %.6f rad/s | theta_ddot = %.6f rad/s^2 | cycles = %d (%.1f%%)\n', ...
        t + dt, theta_meas, theta_dot_meas, theta_ddot_meas, ...
        cycle_count_log(k+1), frac_cycle_log(k+1)*100);

    % ===================================================================
    % SEND ONLY theta AND theta_dot TO C++ (2 doubles instead of 3)
    % ===================================================================
    try
        fwrite(tcpipClient, [theta_meas; theta_dot_meas], 'double');
    catch
        fprintf('\nTCP transmission error\n');
        break;
    end
end

%% =====================================================================
%% SAVE theta_ddot TO CSV (time, theta_ddot_noisy)
%% =====================================================================
csv_data = [tsol(:), theta_ddot_noisy(:)];
csv_filename = 'theta_ddot_log.csv';

fid = fopen(csv_filename, 'w');
fprintf(fid, 'time_s,theta_ddot_noisy_rad_per_s2\n');
fclose(fid);

dlmwrite(csv_filename, csv_data, '-append', 'delimiter', ',', 'precision', '%.9f');

fprintf('\nSaved %d rows of theta_ddot data to "%s"\n', N, csv_filename);
%% =====================================================================

%% Print final cycle summary
total_cycles    = cycle_count_log(end);
total_rot_rad   = abs(theta_cumulative);
total_rot_deg   = total_rot_rad * 180 / pi;
fprintf('\n========================================\n');
fprintf('  MOTION CYCLE SUMMARY\n');
fprintf('========================================\n');
fprintf('  Simulation duration     : %.3f s\n',  tend);
fprintf('  Total crank rotation    : %.4f rad  (%.2f deg)\n', total_rot_rad, total_rot_deg);
fprintf('  Completed full cycles   : %d\n',       total_cycles);
fprintf('  Fractional cycle at end : %.1f%%\n',   frac_cycle_log(end)*100);
fprintf('  Average cycle duration  : %.4f s\n',  ...
        tend / max(total_cycles + frac_cycle_log(end), eps));
fprintf('========================================\n\n');

%% Position, velocity, acceleration plots (True vs Noisy)
figure('Name', 'Input link theta motion (True vs Noisy measurements)');
subplot(3, 1, 1);
plot(tsol, theta_true,  'b-',  'LineWidth', 1.4); hold on;
plot(tsol, theta_noisy, 'r--', 'LineWidth', 0.8);
grid on; ylabel('theta (rad)'); title('Position');
legend('True', 'Noisy (sent to C++)', 'Location', 'best');

subplot(3, 1, 2);
plot(tsol, theta_dot_true,  'b-',  'LineWidth', 1.4); hold on;
plot(tsol, theta_dot_noisy, 'r--', 'LineWidth', 0.8);
grid on; ylabel('theta\_dot (rad/s)'); title('Velocity');
legend('True', 'Noisy (sent to C++)', 'Location', 'best');

subplot(3, 1, 3);
plot(tsol, theta_ddot_true,  'b-',  'LineWidth', 1.4); hold on;
plot(tsol, theta_ddot_noisy, 'r--', 'LineWidth', 0.8);
grid on; ylabel('theta\_ddot (rad/s^2)'); xlabel('t (s)'); title('Acceleration (saved to CSV)');
legend('True', 'Noisy (saved to CSV)', 'Location', 'best');

%% Cleanup
try
    fclose(tcpipClient);
    delete(tcpipClient);
catch
end

%% ===================== functions =====================

function dxdt = fourbarode_lagrange(t, x, p)

    z    = x(1:9);
    zdot = x(10:18);

    M = fourbarMassMatrix(p);
    PhiJac = fourbarConstraintJacobian(z, p);
    Qe = fourbarExternalForces(t, p);
    Qv = zeros(9, 1);

    eps = 1e-6;
    nConstraints = size(PhiJac, 1);
    nCoords      = numel(z);
    dPhiJac_zdot = zeros(nConstraints, nCoords);

    for j = 1:nCoords
        z_plus  = z;  z_minus = z;
        z_plus(j)  = z_plus(j)  + eps;
        z_minus(j) = z_minus(j) - eps;
        PhiJac_plus  = fourbarConstraintJacobian(z_plus,  p);
        PhiJac_minus = fourbarConstraintJacobian(z_minus, p);
        dPhiJac_zdot(:, j) = (PhiJac_plus * zdot - PhiJac_minus * zdot) / (2 * eps);
    end

    gamma = -dPhiJac_zdot * zdot;

    A_aug = [M,      PhiJac';
             PhiJac, zeros(nConstraints, nConstraints)];
    rhs   = [Qe + Qv;
             gamma];

    sol   = A_aug \ rhs;
    zddot = sol(1:9);

    dxdt        = zeros(18, 1);
    dxdt(1:9)   = zdot;
    dxdt(10:18) = zddot;
end


function M = fourbarMassMatrix(p)
    M1 = diag([p.mA, p.mA, p.I1]);
    M2 = diag([p.mB, p.mB, p.I2]);
    M3 = diag([p.mC, p.mC, p.I3]);
    M  = blkdiag(M1, M2, M3);
end


function Qe = fourbarExternalForces(t, p)
    Qe    = zeros(9, 1);
    Qe(2) = -p.mA * p.g;
    Qe(5) = -p.mB * p.g;
    Qe(8) = -p.mC * p.g;
    Qe(3) = Qe(3) + p.T0 * sin(p.omega_input_torque * t);
end


function PhiJac = fourbarConstraintJacobian(z, p)
    theta1 = z(3);
    theta2 = z(6);
    theta3 = z(9);

    L1 = p.L1;  L2 = p.L2;  L3 = p.L3;

    sin1 = sin(theta1);  cos1 = cos(theta1);
    sin2 = sin(theta2);  cos2 = cos(theta2);
    sin3 = sin(theta3);  cos3 = cos(theta3);

    PhiJac = zeros(8, 9);

    PhiJac(1, 1) = 1;    PhiJac(1, 3) =  (L1/2) * sin1;
    PhiJac(2, 2) = 1;    PhiJac(2, 3) = -(L1/2) * cos1;

    PhiJac(3, 1) = 1;    PhiJac(3, 3) = -(L1/2) * sin1;
    PhiJac(3, 4) = -1;   PhiJac(3, 6) = -(L2/2) * sin2;

    PhiJac(4, 2) = 1;    PhiJac(4, 3) =  (L1/2) * cos1;
    PhiJac(4, 5) = -1;   PhiJac(4, 6) =  (L2/2) * cos2;

    PhiJac(5, 4) = 1;    PhiJac(5, 6) = -(L2/2) * sin2;
    PhiJac(5, 7) = -1;   PhiJac(5, 9) =  (L3/2) * sin3;

    PhiJac(6, 5) = 1;    PhiJac(6, 6) =  (L2/2) * cos2;
    PhiJac(6, 8) = -1;   PhiJac(6, 9) = -(L3/2) * cos3;

    PhiJac(7, 7) = 1;    PhiJac(7, 9) =  (L3/2) * sin3;
    PhiJac(8, 8) = 1;    PhiJac(8, 9) = -(L3/2) * cos3;
end


function [B, C, theta2, theta3] = closure_from_theta1(theta1, p)
    A_joint = [0; 0];
    D_joint = [p.L4; 0];

    B = A_joint + [p.L1 * cos(theta1); p.L1 * sin(theta1)];

    d = norm(D_joint - B);
    if d > (p.L2 + p.L3) || d < abs(p.L2 - p.L3)
        error('No real closure');
    end

    ex = (D_joint - B) / d;
    ey = [-ex(2); ex(1)];

    a  = (p.L2^2 - p.L3^2 + d^2) / (2 * d);
    h  = sqrt(max(p.L2^2 - a^2, 0));
    P2 = B + a * ex;
    C1 = P2 + h * ey;
    C2 = P2 - h * ey;

    if C1(2) >= C2(2)
        C = C1;
    else
        C = C2;
    end

    theta2 = atan2(C(2) - B(2), C(1) - B(1));
    theta3 = atan2(C(2) - D_joint(2), C(1) - D_joint(1));
end


function [rA, rB, rC, rD] = joints_from_z(z, p)
    Rx1    = z(1);  Ry1    = z(2);  theta1 = z(3);
    Rx2    = z(4);  Ry2    = z(5);  theta2 = z(6);

    rA = [0; 0];

    rB = [Rx1 + (p.L1/2) * cos(theta1);
          Ry1 + (p.L1/2) * sin(theta1)];

    rC = [Rx2 + (p.L2/2) * cos(theta2);
          Ry2 + (p.L2/2) * sin(theta2)];

    rD = [p.L4; 0];
end
