// ============================================================================
// Pendulum_IKF_Numerical.cpp
// Numerical h_x (Section 3.5.3)
// Compile:
//   g++ -std=c++11 -O3 Pendulum_IKF_Numerical.cpp -o pendulum_server
// Run:
//   ./pendulum_server
// ============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <algorithm>

const double PI = 3.14159265358979323846;

// ====================== STRUCTURES ======================
struct SystemParams {
    double g, L, m, I;   // gravity, length, mass, moment of inertia
    double T0, win;      // optional torque amplitude and frequency
};

struct State {
    double z[3];       // Generalized coordinates [Rx, Ry, theta] (Eq.1)
    double zdot[3];    // Generalized velocities
    double zddot[3];   // Generalized accelerations
};

// IKF error state — Chapter 3, Section 3.1 (Eq.28)
struct IKFState {
    double x_hat[3];      // Error state: [delta_zi, delta_zdoti, delta_zddoti] (Eq.28)
    double P[3][3];       // Error covariance (Eq.30)
    double Sigma_P[3][3]; // Process noise covariance (Eq.32)
    double Sigma_S[2][2]; // Measurement noise covariance (Eq.40) — 2x2 (position + velocity only)
};

struct LogData {
    std::vector<double> time;
    std::vector<double> o_theta,        o_theta_dot,        o_theta_ddot;
    std::vector<double> theta_cpp,      theta_dot_cpp,      theta_ddot_cpp;
    std::vector<double> theta_ikf,      theta_dot_ikf,      theta_ddot_ikf;
    std::vector<double> cycle_time_us;
    std::vector<double> recv_time_us, ikf_time_us, save_time_us, other_time_us;
    // P diagonal elements for 95% confidence interval (±1.96*sqrt(P_ii))
    std::vector<double> P_zi;      // P[0][0]: variance of delta_zi
    std::vector<double> P_zdoti;   // P[1][1]: variance of delta_zdoti
    std::vector<double> P_zddoti;  // P[2][2]: variance of delta_zddoti
};

LogData logData;

// ====================== BYTE SWAP (Section 4.4) ======================
double swapDouble(double value) {
    uint64_t temp;
    std::memcpy(&temp, &value, sizeof(double));
    temp = ((temp & 0xFF00000000000000ULL) >> 56) |
           ((temp & 0x00FF000000000000ULL) >> 40) |
           ((temp & 0x0000FF0000000000ULL) >> 24) |
           ((temp & 0x000000FF00000000ULL) >>  8) |
           ((temp & 0x00000000FF000000ULL) <<  8) |
           ((temp & 0x0000000000FF0000ULL) << 24) |
           ((temp & 0x000000000000FF00ULL) << 40) |
           ((temp & 0x00000000000000FFULL) << 56);
    double result;
    std::memcpy(&result, &temp, sizeof(double));
    return result;
}

// ====================== TCP RECV ALL (Section 4.3) ======================
static ssize_t recvAll(int sockfd, void* buf, size_t len) {
    size_t total = 0;
    char* p = static_cast<char*>(buf);
    while (total < len) {
        ssize_t n = recv(sockfd, p + total, len - total, 0);
        if (n <= 0) return n;
        total += static_cast<size_t>(n);
    }
    return static_cast<ssize_t>(total);
}

// ====================== 3x3 MATRIX UTILITIES ======================
void mult3x3(const double A[3][3], const double B[3][3], double C[3][3]) {
    double tmp[3][3] = {{0}};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                tmp[i][j] += A[i][k] * B[k][j];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            C[i][j] = tmp[i][j];
}

void add3x3(const double A[3][3], const double B[3][3], double C[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            C[i][j] = A[i][j] + B[i][j];
}

void transpose3x3(const double A[3][3], double AT[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            AT[i][j] = A[j][i];
}

// ====================== 5x5 SYSTEM SOLVE (Eq.14a) ======================
bool gaussianElimination5(double A[5][5], double b[5], double sol[5]) {
    double a[5][5];
    double rhs[5];
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) a[i][j] = A[i][j];
        rhs[i] = b[i];
    }
    for (int col = 0; col < 5; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 5; ++row)
            if (std::fabs(a[row][col]) > std::fabs(a[pivot][col]))
                pivot = row;
        if (pivot != col) {
            for (int j = 0; j < 5; ++j) std::swap(a[col][j], a[pivot][j]);
            std::swap(rhs[col], rhs[pivot]);
        }
        if (std::fabs(a[col][col]) < 1e-12) return false;
        for (int row = col + 1; row < 5; ++row) {
            double f = a[row][col] / a[col][col];
            for (int j = col; j < 5; ++j) a[row][j] -= f * a[col][j];
            rhs[row] -= f * rhs[col];
        }
    }
    for (int i = 4; i >= 0; --i) {
        sol[i] = rhs[i];
        for (int j = i + 1; j < 5; ++j) sol[i] -= a[i][j] * sol[j];
        sol[i] /= a[i][i];
    }
    return true;
}

// 2x2 system solve — used for post-correction (Eq.41, 44, 47)
bool gaussianElimination2(double A[2][2], double b[2], double sol[2]) {
    double det = A[0][0]*A[1][1] - A[0][1]*A[1][0];
    if (std::fabs(det) < 1e-12) return false;
    sol[0] = ( A[1][1]*b[0] - A[0][1]*b[1]) / det;
    sol[1] = (-A[1][0]*b[0] + A[0][0]*b[1]) / det;
    return true;
}

// ====================== PENDULUM CONSTRAINTS (Eq.2–3) ======================
void pendulumPhi(const double z[3], const SystemParams& p, double Phi[2]) {
    const double Rx = z[0], Ry = z[1], th = z[2];
    const double Lh = p.L / 2.0;

    // Eq.2: phi1
    Phi[0] = Rx - Lh * std::cos(th);
    // Eq.3: phi2
    Phi[1] = Ry - Lh * std::sin(th);
}

// Constraint Jacobian Phi_z — numerical central differences (Eq.5)
void pendulumPhiz_numerical(const double z[3], const SystemParams& p,
                             double Phi_z[2][3]) {
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            Phi_z[i][j] = 0.0;

    for (int j = 0; j < 3; ++j) {
        double zp[3], zm[3];
        for (int k = 0; k < 3; ++k) { zp[k] = z[k]; zm[k] = z[k]; }
        double h = 1e-6 * (std::fabs(z[j]) + 1.0);
        zp[j] += h; zm[j] -= h;
        double Phi_p[2], Phi_m[2];
        pendulumPhi(zp, p, Phi_p);
        pendulumPhi(zm, p, Phi_m);
        for (int i = 0; i < 2; ++i)
            Phi_z[i][j] = (Phi_p[i] - Phi_m[i]) / (2.0 * h);
    }
}

// Partition Phi_z into independent column (col 2 = theta) and dependent columns
// Used in post-correction (Section 3.4, Eq.41, 44, 47)
void partitionJacobian(const double Phi_z[2][3],
                       double Phi_iz[2],     // 2x1: independent column (theta)
                       double Phi_dz[2][2])  // 2x2: dependent columns
{
    for (int i = 0; i < 2; ++i) Phi_iz[i] = Phi_z[i][2];

    int dep_cols[2] = {0, 1};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            Phi_dz[i][j] = Phi_z[i][dep_cols[j]];
}

// ====================== MASS MATRIX (Eq.7) ======================
void pendulumM(const SystemParams& p, double M[3][3]) {
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) M[i][j] = 0.0;
    M[0][0] = p.m;
    M[1][1] = p.m;
    M[2][2] = p.I;   // Eq.6: I = mL^2/12
}

// ====================== GENERALIZED FORCES (Eq.10a, 9a) ======================
void pendulumQe(double t, const SystemParams& p, double Qe[3]) {
    for (int i = 0; i < 3; ++i) Qe[i] = 0.0;
    Qe[1] = -p.m * p.g;                    // Qe,2 (Eq.10a)
    Qe[2] =  p.T0 * std::sin(p.win * t);
}

// ====================== ACCELERATION SOLVE (Eq.14a) ======================
void computeAccelerations(const State& state, const SystemParams& p,
                          double zddot[3]) {
    const double* z    = state.z;
    const double* zdot = state.zdot;

    double M[3][3];
    pendulumM(p, M);

    double Phi_z[2][3];
    pendulumPhiz_numerical(z, p, Phi_z);

    double Qe[3];
    pendulumQe(0.0, p, Qe);
    double Qv[3] = {0};  // Qv = 0 (Eq.9a)

    // gamma computed numerically (Eq.13, 13a)
    const double eps = 1e-6;
    double dPhiz_dz_zdot[2][3];
    for (int j = 0; j < 3; ++j) {
        double zp[3], zm[3];
        for (int k = 0; k < 3; ++k) { zp[k] = z[k]; zm[k] = z[k]; }
        zp[j] += eps; zm[j] -= eps;
        double Phi_zp[2][3], Phi_zm[2][3];
        pendulumPhiz_numerical(zp, p, Phi_zp);
        pendulumPhiz_numerical(zm, p, Phi_zm);
        for (int i = 0; i < 2; ++i) {
            double cp = 0.0, cm = 0.0;
            for (int k = 0; k < 3; ++k) { cp += Phi_zp[i][k]*zdot[k]; cm += Phi_zm[i][k]*zdot[k]; }
            dPhiz_dz_zdot[i][j] = (cp - cm) / (2.0 * eps);
        }
    }
    double gamma[2];
    for (int i = 0; i < 2; ++i) {
        gamma[i] = 0.0;
        for (int j = 0; j < 3; ++j) gamma[i] -= dPhiz_dz_zdot[i][j] * zdot[j];
    }

    // Build 5x5 augmented system (Eq.14a, expanded in Eq.14)
    double A[5][5], b[5];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) A[i][j] = M[i][j];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 2; ++j) A[i][3+j] = Phi_z[j][i];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 3; ++j) A[3+i][j] = Phi_z[i][j];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) A[3+i][3+j] = 0.0;
    for (int i = 0; i < 3; ++i) b[i] = Qe[i] + Qv[i];
    for (int i = 0; i < 2; ++i) b[3+i] = gamma[i];

    double sol[5];
    if (!gaussianElimination5(A, b, sol)) return;
    for (int i = 0; i < 3; ++i) zddot[i] = sol[i];
}

// ====================== RK4 (Section 4.1, Eq.52) ======================
void rk4Step(State& state, double dt, const SystemParams& p) {
    State k[4];
    double zddot_k[3];

    computeAccelerations(state, p, zddot_k);
    for (int i = 0; i < 3; ++i) { k[0].z[i] = state.zdot[i]; k[0].zdot[i] = zddot_k[i]; }

    State tmp = state;
    for (int i = 0; i < 3; ++i) { tmp.z[i] += 0.5*dt*k[0].z[i]; tmp.zdot[i] += 0.5*dt*k[0].zdot[i]; }
    computeAccelerations(tmp, p, zddot_k);
    for (int i = 0; i < 3; ++i) { k[1].z[i] = tmp.zdot[i]; k[1].zdot[i] = zddot_k[i]; }

    tmp = state;
    for (int i = 0; i < 3; ++i) { tmp.z[i] += 0.5*dt*k[1].z[i]; tmp.zdot[i] += 0.5*dt*k[1].zdot[i]; }
    computeAccelerations(tmp, p, zddot_k);
    for (int i = 0; i < 3; ++i) { k[2].z[i] = tmp.zdot[i]; k[2].zdot[i] = zddot_k[i]; }

    tmp = state;
    for (int i = 0; i < 3; ++i) { tmp.z[i] += dt*k[2].z[i]; tmp.zdot[i] += dt*k[2].zdot[i]; }
    computeAccelerations(tmp, p, zddot_k);
    for (int i = 0; i < 3; ++i) { k[3].z[i] = tmp.zdot[i]; k[3].zdot[i] = zddot_k[i]; }

    for (int i = 0; i < 3; ++i) {
        state.z[i]    += (dt/6.0)*(k[0].z[i]    + 2*k[1].z[i]    + 2*k[2].z[i]    + k[3].z[i]);
        state.zdot[i] += (dt/6.0)*(k[0].zdot[i] + 2*k[1].zdot[i] + 2*k[2].zdot[i] + k[3].zdot[i]);
    }
    computeAccelerations(state, p, state.zddot);
}

// ====================== CLOSURE FROM theta ======================
void closureFromTheta1(double th, const SystemParams& p,
                       double& Rx, double& Ry) {
    const double Lh = p.L / 2.0;
    Rx = Lh * std::cos(th);
    Ry = Lh * std::sin(th);
}

// ====================== IKF — INITIALIZE (Chapter 3) ======================
void ikfInitialize(IKFState& ikf) {
    // x_hat = 0 initially (Eq.28)
    for (int i = 0; i < 3; ++i) ikf.x_hat[i] = 0.0;

    // P: initial error covariance (Eq.30)
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) ikf.P[i][j] = 0.0;
    ikf.P[0][0] = 1e-6;
    ikf.P[1][1] = 1e-6;
    ikf.P[2][2] = 0.1;
    ikf.P[1][2] = 0.0005;
    ikf.P[2][1] = 0.0005;

    // Sigma_P: process noise (Eq.32)
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) ikf.Sigma_P[i][j] = 0.0;
    ikf.Sigma_P[2][2] = 0.02;

    // Sigma_S: measurement noise covariance — 2x2 (Eq.40)
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) ikf.Sigma_S[i][j] = 0.0;
    const double sigma_z    = 0.001;
    const double sigma_zdot = 0.005;
    ikf.Sigma_S[0][0] = sigma_z    * sigma_z;
    ikf.Sigma_S[1][1] = sigma_zdot * sigma_zdot;
}

// Compute acceleration Jacobians numerically for f_bar_x (Eq.31)
void computeAccelJacobians(const State& est, const SystemParams& p,
                            double& dZddot_dZ, double& dZddot_dZdot)
{
    const double eps = 1e-5;
    const double th  = est.z[2];
    const double om  = est.zdot[2];
    const double Lh  = p.L / 2.0;

    auto makeStateFromTheta = [&](double th_p, double om_v) -> State {
        State s;
        s.z[0] = Lh * std::cos(th_p);
        s.z[1] = Lh * std::sin(th_p);
        s.z[2] = th_p;
        s.zdot[0] = -Lh * std::sin(th_p) * om_v;
        s.zdot[1] =  Lh * std::cos(th_p) * om_v;
        s.zdot[2] = om_v;
        for (int i = 0; i < 3; ++i) s.zddot[i] = 0.0;
        return s;
    };

    // d(zddoti)/d(zi) — numerical central difference
    State sp = makeStateFromTheta(th + eps, om);
    State sm = makeStateFromTheta(th - eps, om);
    double zddot_p[3], zddot_m[3];
    computeAccelerations(sp, p, zddot_p);
    computeAccelerations(sm, p, zddot_m);
    dZddot_dZ = (zddot_p[2] - zddot_m[2]) / (2.0 * eps);

    // d(zddoti)/d(zdoti) — numerical central difference
    State sqp = makeStateFromTheta(th, om + eps);
    State sqm = makeStateFromTheta(th, om - eps);
    double zddot_qp[3], zddot_qm[3];
    computeAccelerations(sqp, p, zddot_qp);
    computeAccelerations(sqm, p, zddot_qm);
    dZddot_dZdot = (zddot_qp[2] - zddot_qm[2]) / (2.0 * eps);
}

// ====================== IKF PREDICTION (Section 3.2, Eq.29–32) ======================
void ikfPredict(IKFState& ikf, const State& est,
                const SystemParams& p, double dt) {

    // Acceleration Jacobians for f_bar_x (Eq.31)
    double dZddot_dZ, dZddot_dZdot;
    computeAccelJacobians(est, p, dZddot_dZ, dZddot_dZdot);

    // f_bar_x: discrete state transition matrix (Eq.31)
    double fx[3][3];
    fx[0][0] = 1.0 + 0.5*dZddot_dZ*dt*dt;
    fx[0][1] = dt  + 0.5*dZddot_dZdot*dt*dt;
    fx[0][2] = 0.5*dt*dt;

    fx[1][0] = dZddot_dZ*dt;
    fx[1][1] = 1.0 + dZddot_dZdot*dt;
    fx[1][2] = dt;

    fx[2][0] = 0.0;
    fx[2][1] = 0.0;
    fx[2][2] = 1.0;

    // P_minus = f_bar_x * P_plus * f_bar_x^T + Sigma_P (Eq.30)
    double fx_T[3][3];
    transpose3x3(fx, fx_T);
    double tmp[3][3], P_new[3][3];
    mult3x3(fx, ikf.P, tmp);
    mult3x3(tmp, fx_T, P_new);
    add3x3(P_new, ikf.Sigma_P, ikf.P);

    // x_hat_minus = 0 (Eq.29)
    for (int i = 0; i < 3; ++i) ikf.x_hat[i] = 0.0;
}

// ====================== MEASUREMENT MODEL (Eq.50) ======================
// Nm = 2: only position and velocity measured (no acceleration)
void measurementH(const State& s, double h_out[2]) {
    h_out[0] = s.z[2];     // theta
    h_out[1] = s.zdot[2];  // theta_dot
}

// Numerical hx — Section 3.5.3 (Eq.51)
// hx in R^(2x3): forward finite differences
void computeHx_numerical(const State& est, double hx[2][3]) {
    const double eps_z     = 1e-6;
    const double eps_zdot  = 1e-6;
    const double eps_zddot = 1e-6;

    // h(z, zdot, zddot)
    double h0[2];
    measurementH(est, h0);

    // h_zi (Eq.51)
    State s_zi = est;
    s_zi.z[2] += eps_z;
    double h_zi[2];
    measurementH(s_zi, h_zi);
    hx[0][0] = (h_zi[0] - h0[0]) / eps_z;
    hx[1][0] = (h_zi[1] - h0[1]) / eps_z;

    // h_zdoti (Eq.51)
    State s_zdoti = est;
    s_zdoti.zdot[2] += eps_zdot;
    double h_zdoti[2];
    measurementH(s_zdoti, h_zdoti);
    hx[0][1] = (h_zdoti[0] - h0[0]) / eps_zdot;
    hx[1][1] = (h_zdoti[1] - h0[1]) / eps_zdot;

    // h_zddoti (Eq.51) — h() does not use zddot[2], so this column = 0
    State s_zddoti = est;
    s_zddoti.zddot[2] += eps_zddot;
    double h_zddoti[2];
    measurementH(s_zddoti, h_zddoti);
    hx[0][2] = (h_zddoti[0] - h0[0]) / eps_zddot;
    hx[1][2] = (h_zddoti[1] - h0[1]) / eps_zddot;
}

// ====================== IKF CORRECTION (Section 3.3, Eq.33–37) ======================
// Nm = 2: hx is 2x3, S is 2x2, K is 3x2
void ikfCorrect(IKFState& ikf, const State& est,
                double o_theta, double o_theta_dot,
                const SystemParams& p) {
    (void)p;

    // hx — numerical Jacobian (Section 3.5.3, Eq.51)
    double hx[2][3];
    computeHx_numerical(est, hx);

    // Innovation y = o - h(z_bar, zdot_bar) (Eq.33)
    double h0[2];
    measurementH(est, h0);
    double y[2];
    y[0] = o_theta     - h0[0];
    y[1] = o_theta_dot - h0[1];

    // S = hx * P_minus * hx^T + Sigma_S (Eq.34) — 2x2
    double hxP[2][3];
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c) {
            hxP[r][c] = 0.0;
            for (int k = 0; k < 3; ++k)
                hxP[r][c] += hx[r][k] * ikf.P[k][c];
        }

    double S[2][2];
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c) {
            S[r][c] = ikf.Sigma_S[r][c];
            for (int k = 0; k < 3; ++k)
                S[r][c] += hxP[r][k] * hx[c][k];
        }

    // S inverse (2x2 analytic)
    double det = S[0][0]*S[1][1] - S[0][1]*S[1][0];
    if (std::fabs(det) < 1e-20) return;
    double S_inv[2][2];
    S_inv[0][0] =  S[1][1]/det;
    S_inv[0][1] = -S[0][1]/det;
    S_inv[1][0] = -S[1][0]/det;
    S_inv[1][1] =  S[0][0]/det;

    // K = P_minus * hx^T * S^{-1} (Eq.35) — 3x2
    double PhxT[3][2];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 2; ++c) {
            PhxT[r][c] = 0.0;
            for (int k = 0; k < 3; ++k)
                PhxT[r][c] += ikf.P[r][k] * hx[c][k];
        }

    double K[3][2];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 2; ++c) {
            K[r][c] = 0.0;
            for (int k = 0; k < 2; ++k)
                K[r][c] += PhxT[r][k] * S_inv[k][c];
        }

    // x_hat_plus = 0 + K * y (Eq.36)
    for (int i = 0; i < 3; ++i) {
        ikf.x_hat[i] = 0.0;
        for (int j = 0; j < 2; ++j)
            ikf.x_hat[i] += K[i][j] * y[j];
    }

    // P_plus = (I - K * hx) * P_minus (Eq.37)
    double I_KH[3][3];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            double kh = 0.0;
            for (int k = 0; k < 2; ++k)
                kh += K[r][k] * hx[k][c];
            I_KH[r][c] = (r == c ? 1.0 : 0.0) - kh;
        }

    double P_new[3][3];
    mult3x3(I_KH, ikf.P, P_new);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            ikf.P[r][c] = P_new[r][c];
}

// ====================== POST-CORRECTION (Section 3.4, Eq.41–48) ======================
void ikfApplyCorrection(State& est, const IKFState& ikf,
                        const SystemParams& p) {

    // IKF error state (Eq.28)
    const double delta_zi     = ikf.x_hat[0];
    const double delta_zdoti  = ikf.x_hat[1];
    const double delta_zddoti = ikf.x_hat[2];

    // Corrected independent coordinates
    const double zi_hat     = est.z[2]     + delta_zi;
    const double zdoti_hat  = est.zdot[2]  + delta_zdoti;
    const double zddoti_hat = est.zddot[2] + delta_zddoti;

    // POSITION CORRECTION (Eq.41–42)
    double Rx, Ry;
    closureFromTheta1(zi_hat, p, Rx, Ry);
    est.z[0] = Rx;
    est.z[1] = Ry;
    est.z[2] = zi_hat;

    // VELOCITY CORRECTION (Eq.43–45)
    const double Lh = p.L / 2.0;
    const double s1 = std::sin(zi_hat), c1 = std::cos(zi_hat);
    est.zdot[0] = -Lh * s1 * zdoti_hat;
    est.zdot[1] =  Lh * c1 * zdoti_hat;
    est.zdot[2] = zdoti_hat;

    // ACCELERATION CORRECTION (Eq.46–48)
    double Phi_z[2][3];
    pendulumPhiz_numerical(est.z, p, Phi_z);

    double Phi_iz[2], Phi_dz[2][2];
    partitionJacobian(Phi_z, Phi_iz, Phi_dz);

    // gamma — numerical (Eq.13a)
    const double eps = 1e-6;
    double dPhiz_dz_zdot[2][3];
    for (int j = 0; j < 3; ++j) {
        double zp[3], zm[3];
        for (int k = 0; k < 3; ++k) { zp[k] = est.z[k]; zm[k] = est.z[k]; }
        zp[j] += eps; zm[j] -= eps;
        double Phi_zp[2][3], Phi_zm[2][3];
        pendulumPhiz_numerical(zp, p, Phi_zp);
        pendulumPhiz_numerical(zm, p, Phi_zm);
        for (int i = 0; i < 2; ++i) {
            double cp = 0.0, cm = 0.0;
            for (int k = 0; k < 3; ++k) { cp += Phi_zp[i][k]*est.zdot[k]; cm += Phi_zm[i][k]*est.zdot[k]; }
            dPhiz_dz_zdot[i][j] = (cp - cm) / (2.0 * eps);
        }
    }
    double gamma[2];
    for (int i = 0; i < 2; ++i) {
        gamma[i] = 0.0;
        for (int j = 0; j < 3; ++j) gamma[i] -= dPhiz_dz_zdot[i][j] * est.zdot[j];
    }

    // RHS for dependent accelerations (Eq.47)
    double rhs_acc[2];
    for (int i = 0; i < 2; ++i)
        rhs_acc[i] = gamma[i] - Phi_iz[i] * zddoti_hat;

    // Solve Phi_dz * zddot_d = rhs_acc (Eq.47)
    double zddot_dep[2];
    if (gaussianElimination2(Phi_dz, rhs_acc, zddot_dep)) {
        int dep_cols[2] = {0, 1};
        for (int j = 0; j < 2; ++j)
            est.zddot[dep_cols[j]] = zddot_dep[j];
    }
    // Independent acceleration (Eq.46)
    est.zddot[2] = zddoti_hat;
}

// ====================== CSV SAVE ======================
void saveToCSV() {
    std::ofstream csv("Pendulum_IKF_Numerical.csv");
    csv << "Time,"
           "o_theta,o_theta_dot,o_theta_ddot,"
           "theta_cpp,theta_dot_cpp,theta_ddot_cpp,"
           "theta_ikf,theta_dot_ikf,theta_ddot_ikf,"
           "CycleTime_us,"
           "RecvTime_us,IKFTime_us,SaveTime_us,OtherTime_us,"
           "P_zi,P_zdoti,P_zddoti\n";
    csv << std::fixed << std::setprecision(8);
    for (size_t i = 0; i < logData.time.size(); ++i) {
        csv << logData.time[i]             << ","
            << logData.o_theta[i]          << ","
            << logData.o_theta_dot[i]      << ","
            << logData.o_theta_ddot[i]     << ","
            << logData.theta_cpp[i]        << ","
            << logData.theta_dot_cpp[i]    << ","
            << logData.theta_ddot_cpp[i]   << ","
            << logData.theta_ikf[i]        << ","
            << logData.theta_dot_ikf[i]    << ","
            << logData.theta_ddot_ikf[i]   << ","
            << logData.cycle_time_us[i]    << ","
            << logData.recv_time_us[i]     << ","
            << logData.ikf_time_us[i]      << ","
            << logData.save_time_us[i]     << ","
            << logData.other_time_us[i]    << ","
            << logData.P_zi[i]             << ","
            << logData.P_zdoti[i]          << ","
            << logData.P_zddoti[i]         << "\n";
    }
    csv.close();
    std::cout << "\nSaved " << logData.time.size()
              << " samples to Pendulum_IKF_Numerical.csv\n";
}

// ====================== MAIN (Chapter 4) ======================
int main() {
    const double g   = 9.81;
    const double L   = 1.0;
    const double w   = 0.1, hh = 0.1;
    const double rho = 7850.0;
    const double m   = rho * L * w * hh;
    auto rodI = [](double m, double L){ return m*L*L/12.0; };  // Eq.6

    SystemParams p;
    p.g  = g;
    p.L  = L;
    p.m  = m;
    p.I  = rodI(m, L);
    p.T0 = 0.0;
    p.win = 2.0 * PI * 0.02;

    // TCP/IP server (Section 4.5)
    const int serverPort = 5556;
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { std::cerr << "Socket creation failed\n"; return 1; }
    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr;
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(serverPort);
    addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "Bind failed\n"; return 1; }
    listen(sock, 1);
    std::cout << "Waiting for MATLAB client on port " << serverPort << "...\n";
    int client = accept(sock, NULL, NULL);
    if (client < 0) { std::cerr << "Accept failed\n"; return 1; }
    std::cout << "MATLAB connected!\n";

    // Initial conditions (Eq.1)
    const double th1_0 = -PI / 3.0;
    double Rx0, Ry0;
    closureFromTheta1(th1_0, p, Rx0, Ry0);

    State plant, est;
    plant.z[0] = Rx0;  plant.z[1] = Ry0;  plant.z[2] = th1_0;
    plant.zdot[0] = 0.0; plant.zdot[1] = 0.0; plant.zdot[2] = 0.3;
    for (int i = 0; i < 3; ++i) plant.zddot[i] = 0.0;
    est = plant;

    // Compute initial accelerations (Eq.14a)
    computeAccelerations(plant, p, plant.zddot);
    computeAccelerations(est,   p, est.zddot);

    IKFState ikf;
    ikfInitialize(ikf);

    const double dt_tcp   = 0.001;   // 1 kHz (Section 4.1)
    const int    Nsamples = 10000;
    int    samplecount    = 0;
    double max_cycle_time = 0.0;

    while (samplecount < Nsamples) {
        auto step_start = std::chrono::high_resolution_clock::now();

        // ---- RECEIVE ----
        double o_theta_recv, o_theta_dot_recv, o_theta_ddot_recv;
        if (recvAll(client, &o_theta_recv,    sizeof(double)) <= 0 ||
            recvAll(client, &o_theta_dot_recv,  sizeof(double)) <= 0 ||
            recvAll(client, &o_theta_ddot_recv, sizeof(double)) <= 0) {
            std::cout << "\nConnection lost.\n"; break;
        }
        o_theta_recv     = swapDouble(o_theta_recv);
        o_theta_dot_recv = swapDouble(o_theta_dot_recv);
        o_theta_ddot_recv= swapDouble(o_theta_ddot_recv);

        auto t_after_recv = std::chrono::high_resolution_clock::now();

        samplecount++;
        double t = samplecount * dt_tcp;

        // ---- IKF COMPUTATION (plant RK4 + predict + propagate + correct + post-correct) ----
        // 1) Plant integration — comparison reference (Eq.52)
        rk4Step(plant, dt_tcp, p);
        const double theta_cpp_val     = plant.z[2];
        const double theta_dot_cpp_val = plant.zdot[2];
        const double theta_ddot_cpp_val= plant.zddot[2];

        // 2) MATLAB measurements o = [theta, theta_dot, theta_ddot]
        const double o_theta      = o_theta_recv;
        const double o_theta_dot  = o_theta_dot_recv;
        const double o_theta_ddot = o_theta_ddot_recv;   // logged only, NOT fused

        // 3) IKF PREDICTION (Section 3.2, Eq.29–32)
        ikfPredict(ikf, est, p, dt_tcp);

        // 4) Propagate nominal trajectory to time k+1 (Eq.52)
        rk4Step(est, dt_tcp, p);

        // 5) IKF CORRECTION (Section 3.3, Eq.33–37) — Nm=2, no acceleration measurement
        ikfCorrect(ikf, est, o_theta, o_theta_dot, p);

        // 6) POST-CORRECTION (Section 3.4, Eq.41–48)
        ikfApplyCorrection(est, ikf, p);

        // 7) Extract IKF estimates
        const double theta_ikf_val     = est.z[2];
        const double theta_dot_ikf_val = est.zdot[2];
        const double theta_ddot_ikf_val= est.zddot[2];

        auto t_after_ikf = std::chrono::high_resolution_clock::now();

        // ---- DATA SAVING ----
        double error = o_theta - theta_ikf_val;

        logData.time.push_back(t);
        logData.o_theta.push_back(o_theta);
        logData.o_theta_dot.push_back(o_theta_dot);
        logData.o_theta_ddot.push_back(o_theta_ddot);
        logData.theta_cpp.push_back(theta_cpp_val);
        logData.theta_dot_cpp.push_back(theta_dot_cpp_val);
        logData.theta_ddot_cpp.push_back(theta_ddot_cpp_val);
        logData.theta_ikf.push_back(theta_ikf_val);
        logData.theta_dot_ikf.push_back(theta_dot_ikf_val);
        logData.theta_ddot_ikf.push_back(theta_ddot_ikf_val);

        auto t_after_save = std::chrono::high_resolution_clock::now();

        // ---- TIMING CALCULATION ----
        auto step_end = std::chrono::high_resolution_clock::now();
        double cycle_us = std::chrono::duration<double,std::micro>(step_end-step_start).count();
        double recv_us  = std::chrono::duration<double,std::micro>(t_after_recv-step_start).count();
        double ikf_us   = std::chrono::duration<double,std::micro>(t_after_ikf-t_after_recv).count();
        double save_us  = std::chrono::duration<double,std::micro>(t_after_save-t_after_ikf).count();
        double other_us = cycle_us - recv_us - ikf_us - save_us;
        max_cycle_time  = std::max(max_cycle_time, cycle_us);

        logData.cycle_time_us.push_back(cycle_us);
        logData.recv_time_us.push_back(recv_us);
        logData.ikf_time_us.push_back(ikf_us);
        logData.save_time_us.push_back(save_us);
        logData.other_time_us.push_back(other_us);
        // P diagonal elements (after correction — P_plus)
        logData.P_zi.push_back(ikf.P[0][0]);
        logData.P_zdoti.push_back(ikf.P[1][1]);
        logData.P_zddoti.push_back(ikf.P[2][2]);

        if (samplecount % 1000 == 0) {
            printf("%8.3f %10.6f %10.6f %10.6f"
                   " %10.6f %10.6f %10.6f"
                   " %8.4f %8.4f %8.4f"
                   "  %8.5f  %10.8f  %8.2f us\n",
                   t,
                   o_theta,         theta_cpp_val,      theta_ikf_val,
                   o_theta_dot,     theta_dot_cpp_val,   theta_dot_ikf_val,
                   o_theta_ddot,    theta_ddot_cpp_val,  theta_ddot_ikf_val,
                   ikf.x_hat[2],
                   error, cycle_us);
        }
    }

    close(client);
    close(sock);
    saveToCSV();
    std::cout << "Max cycle time: " << max_cycle_time << " us\n";
    return 0;
}
