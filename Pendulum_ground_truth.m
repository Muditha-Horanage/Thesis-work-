%% AUGMENTED LAGRANGE SIMPLE PENDULUM  WITH TCP/IP AND SENSOR NOISE
% Modified: send only theta and theta_dot to C++; save theta_ddot to CSV
clear; clc; close all;

%% SYSTEM PARAMETERS
% Rod geometry and material
L = 1.0;      % Rod length [m]
w = 0.1;      % Rod width [m]
h = 0.1;      % Rod height [m]
rho = 7850;   % Material density [kg/m^3]
g = 9.81;        % Gravitational acceleration [m/s^2]

% Mass of the rod and its moment of inertia about the centre of mass
m_A = rho * L * w * h;              % Mass of the rod
I_theta = m_A * L^2 / 12;           % Mass moment of inertia (Eq.6)

%% SENSOR NOISE PARAMETERS
% Standard deviations of white Gaussian noise applied to each measured state
% (Eq.40)
sigma_theta      = 1e-3;   % Position noise standard deviation [rad]
sigma_theta_dot  = 1e-2;   % Velocity noise standard deviation [rad/s]
sigma_theta_ddot = 0.05;   % Acceleration noise standard deviation [rad/s^2]

% Generate white Gaussian noise sequences
rng(42);
noise_theta      = sigma_theta      * randn(1, 10000);
noise_theta_dot  = sigma_theta_dot  * randn(1, 10000);
noise_theta_ddot = sigma_theta_ddot * randn(1, 10000);

fprintf('===============================================\n');
fprintf('AUGMENTED LAGRANGE - SIMPLE PENDULUM (SENSOR NOISE)\n');
fprintf('===============================================\n');
fprintf('Length L = %.3f m\n', L);
fprintf('Mass m_A = %.3f kg\n', m_A);
fprintf('Inertia I_theta = %.6f kg*m^2\n', I_theta);
fprintf('Gravity g = %.2f m/s^2\n', g);
fprintf('\nSENSOR NOISE:\n');
fprintf('  sigma_theta      = %.1e rad (%.3f deg)\n', sigma_theta, sigma_theta*180/pi);
fprintf('  sigma_theta_dot  = %.1e rad/s\n', sigma_theta_dot);
fprintf('  sigma_theta_ddot = %.3f rad/s^2\n', sigma_theta_ddot);
fprintf('===============================================\n\n');

%% SIMULATION PARAMETERS
dt    = 0.001;      % Integration step size delta_t = 1 ms (Section 4.1)
t_end = 10;
t     = 0:dt:t_end;
N     = length(t);

%% INITIAL CONDITIONS
% Generalized coordinate vector z = [Rx; Ry; theta] (Eq.1)
z0    = [0.1; 0; -pi/6];   % Initial generalized coordinates [m, m, rad]
zdot0 = [0; 0; 0.5];       % Initial generalized velocities [m/s, m/s, rad/s]

% Full state vector x = [z; zdot]
x0 = [z0; zdot0];

%% TCP/IP CONFIGURATION (Section 4.5)
serverIP   = '169.254.131.136';
serverPort = 5000;

%% CONNECT TO C++ SERVER (Section 4.3)
try
    tcpipClient = tcpip(serverIP, serverPort, 'NetworkRole', 'client');
    tcpipClient.InputBufferSize  = 8192;
    tcpipClient.OutputBufferSize = 8192;
    tcpipClient.Timeout          = 30;
    fopen(tcpipClient);
    fprintf('Connected to C++ server!\n\n');
catch ME
    fprintf('Connection failed: %s\n', ME.message);
    return;
end

%% STORAGE ARRAYS
x               = zeros(6, N);  % True state vector: [z; zdot]
x_noisy         = zeros(6, N);  % Noisy (measured) state vector
theta_ddot_true_all  = zeros(1, N); % True theta_ddot at each step
theta_ddot_noisy_all = zeros(1, N); % Noisy theta_ddot measurement

x(:,1)      = x0;
x_noisy(:,1) = x0;
noise_idx   = 1;

%% SIMULATION LOOP (RK4 + TCP transmission + sensor noise)
% Fourth-order Runge-Kutta (Eq.52)
fprintf('===============================================\n');
fprintf('AUGMENTED LAGRANGE SIMULATION STARTED (with NOISE)\n');
fprintf('===============================================\n');

for k = 1:N-1

    % RK4 slope increments k1 to k4 (Eq.52)
    k1 = augmentedLagrangeODE(x(:,k),         m_A, g, L, I_theta);
    k2 = augmentedLagrangeODE(x(:,k) + dt/2*k1, m_A, g, L, I_theta);
    k3 = augmentedLagrangeODE(x(:,k) + dt/2*k2, m_A, g, L, I_theta);
    k4 = augmentedLagrangeODE(x(:,k) + dt*k3,   m_A, g, L, I_theta);

    % Weighted summation to advance state (Eq.52)
    x_next   = x(:,k) + dt/6 * (k1 + 2*k2 + 2*k3 + k4);
    x(:,k+1) = x_next;

    % True theta_ddot at current step from equations of motion (Eq.14a)
    dxdt            = augmentedLagrangeODE(x(:,k), m_A, g, L, I_theta);
    theta_ddot_true = dxdt(6);

    % Extract theta and theta_dot from state vector (Eq.1)
    theta_true     = x_next(3);
    theta_dot_true = x_next(6);

    % Apply sensor noise to measurements (Eq.40)
    theta_meas      = theta_true     + noise_theta(noise_idx);
    theta_dot_meas  = theta_dot_true + noise_theta_dot(noise_idx);
    theta_ddot_meas = theta_ddot_true + noise_theta_ddot(noise_idx);

    % Store noisy measurements
    x_noisy(3, k+1)          = theta_meas;
    x_noisy(6, k+1)          = theta_dot_meas;
    theta_ddot_true_all(k+1)  = theta_ddot_true;
    theta_ddot_noisy_all(k+1) = theta_ddot_meas;

    % Advance noise index with wrap-around
    noise_idx = noise_idx + 1;
    if noise_idx > length(noise_theta)
        noise_idx = 1;
    end

    % ===================================================================
    % TCP/IP: SEND ONLY theta AND theta_dot TO C++ (2 doubles)
    % ===================================================================
    try
        fwrite(tcpipClient, theta_meas,     'double');  % Send theta
        fwrite(tcpipClient, theta_dot_meas, 'double');  % Send theta_dot

        if mod(k, 1000) == 0
            fprintf('Step %d: theta=%.4f/%.4f rad, theta_dot=%.4f/%.4f rad/s, theta_ddot=%.4f/%.4f rad/s^2\n', ...
                k, theta_meas, theta_true, theta_dot_meas, theta_dot_true, ...
                theta_ddot_meas, theta_ddot_true);
        end
    catch
        fprintf('\nTCP transmission error at step %d\n', k);
        break;
    end
end

%% CLEANUP
try
    fclose(tcpipClient);
    delete(tcpipClient);
    fprintf('\nTCP connection closed\n');
catch
end

%% =====================================================================
%% SAVE theta_ddot TO CSV (time, theta_ddot_noisy)
%% =====================================================================
theta_ddot_true_all(1) = 0;

csv_data = [t(:), theta_ddot_noisy_all(:)];
csv_filename = 'theta_ddot_log.csv';

fid = fopen(csv_filename, 'w');
fprintf(fid, 'time_s,theta_ddot_noisy_rad_per_s2\n');
fclose(fid);

dlmwrite(csv_filename, csv_data, '-append', 'delimiter', ',', 'precision', '%.9f');

fprintf('\nSaved %d rows of theta_ddot data to "%s"\n', N, csv_filename);
%% =====================================================================

%% PLOT: theta, theta_dot, theta_ddot (TRUE vs NOISY)
figure('Position', [100 100 800 600])

subplot(3,1,1)
plot(t, x(3,:),        'b-',  'LineWidth', 2,   'DisplayName', 'True theta')
hold on
plot(t, x_noisy(3,:),  'r--', 'LineWidth', 1.5, 'DisplayName', 'Noisy theta measurement')
grid on
legend('Location', 'best')
xlabel('Time [s]')
ylabel('theta [rad]')
title('Angular Position: True vs Noisy Measurements')

subplot(3,1,2)
plot(t, x(6,:),        'b-',  'LineWidth', 2,   'DisplayName', 'True theta\_dot')
hold on
plot(t, x_noisy(6,:),  'r--', 'LineWidth', 1.5, 'DisplayName', 'Noisy theta\_dot measurement')
grid on
legend('Location', 'best')
xlabel('Time [s]')
ylabel('theta\_dot [rad/s]')
title('Angular Velocity: True vs Noisy Measurements')

subplot(3,1,3)
plot(t, theta_ddot_true_all,  'b-',  'LineWidth', 2,   'DisplayName', 'True theta\_ddot')
hold on
plot(t, theta_ddot_noisy_all, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Noisy theta\_ddot (saved to CSV)')
grid on
legend('Location', 'best')
xlabel('Time [s]')
ylabel('theta\_ddot [rad/s^2]')
title('Angular Acceleration: True vs Noisy Measurements (saved to CSV)')

%% NOISE STATISTICS
fprintf('\n=========================================\n');
fprintf('NOISE STATISTICS:\n');
fprintf('=========================================\n');
fprintf('  theta noise:      RMS=%.3e rad,     max=%.3e rad\n', ...
        rms(noise_theta(1:N-1)),      max(abs(noise_theta(1:N-1))));
fprintf('  theta_dot noise:  RMS=%.3e rad/s,   max=%.3e rad/s\n', ...
        rms(noise_theta_dot(1:N-1)),  max(abs(noise_theta_dot(1:N-1))));
fprintf('  theta_ddot noise: RMS=%.3e rad/s^2, max=%.3e rad/s^2\n', ...
        rms(noise_theta_ddot(1:N-1)), max(abs(noise_theta_ddot(1:N-1))));
fprintf('=========================================\n\n');

%% PENDULUM ANIMATION (using true states for visualization)
figure
hold on
grid on
axis equal

xlim([-L  L])
ylim([-L  0.2])

xlabel('x [m]')
ylabel('y [m]')

h_rod   = plot([0 0], [0 0], 'k',  'LineWidth', 5);
h_bob   = plot(0, 0, 'ko', 'MarkerSize', 18, 'MarkerFaceColor', [1 0.5 0]);
h_pivot = plot(0, 0, 'rs', 'MarkerSize', 12, 'MarkerFaceColor', 'r');

h_trail = animatedline('Color', [0.3 0.3 0.8], 'LineWidth', 1.5);

title('Augmented Lagrange Pendulum (True Motion)')

%% ANIMATION LOOP
frame_skip = 10;

for k = 1:frame_skip:N

    theta = x(3, k);  % True angular position theta (Eq.1)

    % Tip position of the rod
    x_bob = (L/2) * cos(theta);
    y_bob = (L/2) * sin(theta);

    set(h_rod,  'XData', [0 x_bob], 'YData', [0 y_bob])
    set(h_bob,  'XData', x_bob,     'YData', y_bob)

    addpoints(h_trail, x_bob, y_bob)

    title(sprintf('t = %.2f s | angle = %.2f deg', t(k), rad2deg(theta)))

    drawnow

end

%% DISPLAY SYSTEM PARAMETERS
fprintf('\n=========================================\n');
fprintf('PENDULUM SYSTEM PARAMETERS:\n');
fprintf('=========================================\n');
fprintf('Rod length L:              %.2f m\n', L);
fprintf('Mass m_A:                  %.3f kg\n', m_A);
fprintf('Moment of inertia I_theta: %.6f kg*m^2\n', I_theta);
fprintf('Gravity g:                 %.1f m/s^2\n', g);
fprintf('Initial angle theta_0:     %.2f rad (%.1f deg)\n', z0(3), rad2deg(z0(3)));
fprintf('Initial angular velocity:  %.2f rad/s\n', zdot0(3));
fprintf('=========================================\n\n');


%% ==========================================
% AUGMENTED LAGRANGE ODE FUNCTION
% Section 2.1 (Eq.1-14)
% ==========================================
function dxdt = augmentedLagrangeODE(x, m_A, g, L, I_theta)

% Unpack state vector x = [z; zdot]
% Generalized coordinates z = [Rx; Ry; theta] (Eq.1)
z    = x(1:3);
zdot = x(4:6);

theta    = z(3);     % Rotation angle theta (Eq.1)
thetadot = zdot(3);  % Angular velocity theta_dot

%% MASS MATRIX (Eq.7)
M = diag([m_A, m_A, I_theta]);

%% CONSTRAINT JACOBIAN Phi_z (Eq.5)
% Derived from constraint equations (Eq.2-3)
Phi_z = [1,  0,  (L/2)*sin(theta);
         0,  1, -(L/2)*cos(theta)];

%% EXTERNAL GENERALIZED FORCES Qe (Eq.10a)
Q_e = [0; -m_A*g; 0];

%% VELOCITY-DEPENDENT QUADRATIC FORCES Qv = 0 (Eq.9a)
Q_v = [0; 0; 0];

%% TOTAL GENERALIZED FORCE VECTOR Q (Eq.11)
Q_total = Q_e + Q_v;

%% CONSTRAINT VECTOR gamma (Eq.13a)
gamma = [(L/2) * thetadot^2 * cos(theta);
         (L/2) * thetadot^2 * sin(theta)];

%% AUGMENTED LAGRANGIAN SYSTEM (Eq.14a, expanded in Eq.14)
A_sys = [M,     Phi_z';
         Phi_z, zeros(2)];

b_sys = [Q_total;
         -gamma];

% Solve for zddot and lambda (Eq.14a)
sol = A_sys \ b_sys;

zddot = sol(1:3);  % Generalized acceleration vector zddot

% State derivative: [zdot; zddot] (Eq.52)
dxdt = [zdot; zddot];

end
