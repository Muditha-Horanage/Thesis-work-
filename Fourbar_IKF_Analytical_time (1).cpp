// ============================================================================
// Fourbar_IKF_Analytical.cpp
//
// Compile:
//   g++ -std=c++11 -O3 Fourbar_IKF_Analytical1.cpp -o fourbar_server
// Run:
//   ./fourbar_server
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
    double g, L1, L2, L3, L4;
    double m1, m2, m3;
    double I1, I2, I3;
    double T0, win;
};

struct State {
    double z[9];       // Generalized coordinates (Eq.15)
    double zdot[9];    // Generalized velocities
    double zddot[9];   // Generalized accelerations
};

// IKF error state — Chapter 3, Section 3.1 (Eq.28)
struct IKFState {
    double x_hat[3];      // Error state: [delta_zi, delta_zdoti, delta_zddoti] (Eq.28)
    double P[3][3];       // Error covariance (Eq.30)
    double Sigma_P[3][3]; // Process noise covariance (Eq.32)
    double Sigma_S[3][3]; // Measurement noise covariance (Eq.40)
};

struct LogData {
    std::vector<double> time;
    std::vector<double> o_theta,        o_theta_dot,        o_theta_ddot;
    std::vector<double> theta_cpp,      theta_dot_cpp,      theta_ddot_cpp;
    std::vector<double> theta_ikf,      theta_dot_ikf,      theta_ddot_ikf;
    std::vector<double> cycle_time_us;
    std::vector<double> t_recv_us, t_ikf_us, t_save_us, t_other_us;
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

// ====================== 3x3 MATRIX INVERSE ======================
bool invert3x3(const double A[3][3], double Ainv[3][3]) {
    double det = A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1])
               - A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0])
               + A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]);
    if (std::fabs(det) < 1e-20) return false;
    double invDet = 1.0 / det;
    Ainv[0][0] =  (A[1][1]*A[2][2] - A[1][2]*A[2][1]) * invDet;
    Ainv[0][1] = -(A[0][1]*A[2][2] - A[0][2]*A[2][1]) * invDet;
    Ainv[0][2] =  (A[0][1]*A[1][2] - A[0][2]*A[1][1]) * invDet;
    Ainv[1][0] = -(A[1][0]*A[2][2] - A[1][2]*A[2][0]) * invDet;
    Ainv[1][1] =  (A[0][0]*A[2][2] - A[0][2]*A[2][0]) * invDet;
    Ainv[1][2] = -(A[0][0]*A[1][2] - A[0][2]*A[1][0]) * invDet;
    Ainv[2][0] =  (A[1][0]*A[2][1] - A[1][1]*A[2][0]) * invDet;
    Ainv[2][1] = -(A[0][0]*A[2][1] - A[0][1]*A[2][0]) * invDet;
    Ainv[2][2] =  (A[0][0]*A[1][1] - A[0][1]*A[1][0]) * invDet;
    return true;
}

// ====================== 17x17 SYSTEM SOLVE (Eq.14b) ======================
bool gaussianElimination17(double A[17][17], double b[17], double sol[17]) {
    double a[17][17];
    double rhs[17];
    for (int i = 0; i < 17; ++i) {
        for (int j = 0; j < 17; ++j) a[i][j] = A[i][j];
        rhs[i] = b[i];
    }
    for (int col = 0; col < 17; ++col) {
        int pivot = col;
        for (int row = col+1; row < 17; ++row)
            if (std::fabs(a[row][col]) > std::fabs(a[pivot][col]))
                pivot = row;
        if (pivot != col) {
            for (int j = 0; j < 17; ++j) std::swap(a[col][j], a[pivot][j]);
            std::swap(rhs[col], rhs[pivot]);
        }
        if (std::fabs(a[col][col]) < 1e-12) return false;
        for (int row = col+1; row < 17; ++row) {
            double f = a[row][col] / a[col][col];
            for (int j = col; j < 17; ++j) a[row][j] -= f * a[col][j];
            rhs[row] -= f * rhs[col];
        }
    }
    for (int i = 16; i >= 0; --i) {
        sol[i] = rhs[i];
        for (int j = i+1; j < 17; ++j) sol[i] -= a[i][j] * sol[j];
        sol[i] /= a[i][i];
    }
    return true;
}

// 8x8 system solve — used for post-correction (Eq.41, 44, 47)
bool gaussianElimination8(double A[8][8], double b[8], double sol[8]) {
    double a[8][8];
    double rhs[8];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) a[i][j] = A[i][j];
        rhs[i] = b[i];
    }
    for (int col = 0; col < 8; ++col) {
        int pivot = col;
        for (int row = col+1; row < 8; ++row)
            if (std::fabs(a[row][col]) > std::fabs(a[pivot][col]))
                pivot = row;
        if (pivot != col) {
            for (int j = 0; j < 8; ++j) std::swap(a[col][j], a[pivot][j]);
            std::swap(rhs[col], rhs[pivot]);
        }
        if (std::fabs(a[col][col]) < 1e-12) return false;
        for (int row = col+1; row < 8; ++row) {
            double f = a[row][col] / a[col][col];
            for (int j = col; j < 8; ++j) a[row][j] -= f * a[col][j];
            rhs[row] -= f * rhs[col];
        }
    }
    for (int i = 7; i >= 0; --i) {
        sol[i] = rhs[i];
        for (int j = i+1; j < 8; ++j) sol[i] -= a[i][j] * sol[j];
        sol[i] /= a[i][i];
    }
    return true;
}

// ====================== FOUR-BAR CONSTRAINTS (Eq.16–23) ======================
void fourbarPhi(const double z[9], const SystemParams& p, double Phi[8]) {
    const double Rx1=z[0], Ry1=z[1], th1=z[2];
    const double Rx2=z[3], Ry2=z[4], th2=z[5];
    const double Rx3=z[6], Ry3=z[7], th3=z[8];
    const double L1h=p.L1/2.0, L2h=p.L2/2.0, L3h=p.L3/2.0;

    // Eq.16–17: ground pin A=(0,0) to link1 centre of mass
    Phi[0] = Rx1 - L1h*std::cos(th1);
    Phi[1] = Ry1 - L1h*std::sin(th1);

    // Eq.18–19: joint B — link1 end = link2 start
    Phi[2] = (Rx1 + L1h*std::cos(th1)) - (Rx2 - L2h*std::cos(th2));
    Phi[3] = (Ry1 + L1h*std::sin(th1)) - (Ry2 - L2h*std::sin(th2));

    // Eq.20–21: joint C — link2 end = link3 start
    Phi[4] = (Rx2 + L2h*std::cos(th2)) - (Rx3 + L3h*std::cos(th3));
    Phi[5] = (Ry2 + L2h*std::sin(th2)) - (Ry3 + L3h*std::sin(th3));

    // Eq.22–23: ground pin D=(L4,0) to link3 centre of mass
    Phi[6] = Rx3 - (p.L4 + L3h*std::cos(th3));
    Phi[7] = Ry3 -          L3h*std::sin(th3);
}

// Constraint Jacobian Phi_z — numerical central differences (Eq.24a)
void fourbarPhiz_numerical(const double z[9], const SystemParams& p,
                            double Phi_z[8][9]) {
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 9; ++j)
            Phi_z[i][j] = 0.0;

    for (int j = 0; j < 9; ++j) {
        double zp[9], zm[9];
        for (int k = 0; k < 9; ++k) { zp[k]=z[k]; zm[k]=z[k]; }
        double h = 1e-6*(std::fabs(z[j])+1.0);
        zp[j] += h; zm[j] -= h;
        double Phi_p[8], Phi_m[8];
        fourbarPhi(zp, p, Phi_p);
        fourbarPhi(zm, p, Phi_m);
        for (int i = 0; i < 8; ++i)
            Phi_z[i][j] = (Phi_p[i]-Phi_m[i])/(2.0*h);
    }
}

// Partition Phi_z into independent column (col 2 = theta1) and dependent columns
// Used in post-correction (Section 3.4, Eq.41, 44, 47)
void partitionJacobian(const double Phi_z[8][9],
                       double Phi_iz[8],     // 8x1: independent column (theta1)
                       double Phi_dz[8][8])  // 8x8: dependent columns
{
    for (int i = 0; i < 8; ++i) Phi_iz[i] = Phi_z[i][2];

    int dep_cols[8] = {0,1,3,4,5,6,7,8};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            Phi_dz[i][j] = Phi_z[i][dep_cols[j]];
}

// ====================== MASS MATRIX (Eq.26–27) ======================
void fourbarM(const SystemParams& p, double M[9][9]) {
    for (int i=0;i<9;++i) for (int j=0;j<9;++j) M[i][j]=0.0;
    M[0][0]=p.m1; M[1][1]=p.m1; M[2][2]=p.I1;
    M[3][3]=p.m2; M[4][4]=p.m2; M[5][5]=p.I2;
    M[6][6]=p.m3; M[7][7]=p.m3; M[8][8]=p.I3;
}

// ====================== GENERALIZED FORCES (Eq.10b, 9b) ======================
void fourbarQe(double t, const SystemParams& p, double Qe[9]) {
    for (int i=0;i<9;++i) Qe[i]=0.0;
    Qe[1]=-p.m1*p.g;   // Qe,2 (Eq.10b)
    Qe[4]=-p.m2*p.g;   // Qe,5 (Eq.10b)
    Qe[7]=-p.m3*p.g;   // Qe,8 (Eq.10b)
    Qe[2]= p.T0*std::sin(p.win*t);
}

// ====================== ACCELERATION SOLVE (Eq.14b) ======================
void computeAccelerations(const State& state, const SystemParams& p,
                           double zddot[9]) {
    const double* z    = state.z;
    const double* zdot = state.zdot;

    double M[9][9];
    fourbarM(p, M);

    double Phi_z[8][9];
    fourbarPhiz_numerical(z, p, Phi_z);

    double Qe[9];
    fourbarQe(0.0, p, Qe);
    double Qv[9] = {0};  // Qv = 0 (Eq.9b)

    // gamma = -(Phi_z_dot * zdot) computed numerically (Eq.13, 13b)
    const double eps = 1e-6;
    double dPhiz_dz_zdot[8][9];
    for (int j = 0; j < 9; ++j) {
        double zp[9], zm[9];
        for (int k=0;k<9;++k){zp[k]=z[k];zm[k]=z[k];}
        zp[j]+=eps; zm[j]-=eps;
        double Phi_zp[8][9], Phi_zm[8][9];
        fourbarPhiz_numerical(zp,p,Phi_zp);
        fourbarPhiz_numerical(zm,p,Phi_zm);
        for (int i=0;i<8;++i){
            double cp=0.0,cm=0.0;
            for(int k=0;k<9;++k){cp+=Phi_zp[i][k]*zdot[k]; cm+=Phi_zm[i][k]*zdot[k];}
            dPhiz_dz_zdot[i][j]=(cp-cm)/(2.0*eps);
        }
    }
    double gamma[8];
    for (int i=0;i<8;++i){
        gamma[i]=0.0;
        for (int j=0;j<9;++j) gamma[i]-=dPhiz_dz_zdot[i][j]*zdot[j];
    }

    // Build 17x17 augmented system (Eq.14b)
    double A[17][17], b[17];
    for(int i=0;i<9;++i) for(int j=0;j<9;++j) A[i][j]=M[i][j];
    for(int i=0;i<9;++i) for(int j=0;j<8;++j) A[i][9+j]=Phi_z[j][i];
    for(int i=0;i<8;++i) for(int j=0;j<9;++j) A[9+i][j]=Phi_z[i][j];
    for(int i=0;i<8;++i) for(int j=0;j<8;++j) A[9+i][9+j]=0.0;
    for(int i=0;i<9;++i) b[i]=Qe[i]+Qv[i];
    for(int i=0;i<8;++i) b[9+i]=gamma[i];

    double sol[17];
    if(!gaussianElimination17(A,b,sol)) return;
    for(int i=0;i<9;++i) zddot[i]=sol[i];
}

// ====================== RK4 (Section 4.1, Eq.52) ======================
void rk4Step(State& state, double dt, const SystemParams& p) {
    State k[4];
    double zddot_k[9];

    computeAccelerations(state, p, zddot_k);
    for(int i=0;i<9;++i){k[0].z[i]=state.zdot[i]; k[0].zdot[i]=zddot_k[i];}

    State tmp=state;
    for(int i=0;i<9;++i){tmp.z[i]+=0.5*dt*k[0].z[i]; tmp.zdot[i]+=0.5*dt*k[0].zdot[i];}
    computeAccelerations(tmp,p,zddot_k);
    for(int i=0;i<9;++i){k[1].z[i]=tmp.zdot[i]; k[1].zdot[i]=zddot_k[i];}

    tmp=state;
    for(int i=0;i<9;++i){tmp.z[i]+=0.5*dt*k[1].z[i]; tmp.zdot[i]+=0.5*dt*k[1].zdot[i];}
    computeAccelerations(tmp,p,zddot_k);
    for(int i=0;i<9;++i){k[2].z[i]=tmp.zdot[i]; k[2].zdot[i]=zddot_k[i];}

    tmp=state;
    for(int i=0;i<9;++i){tmp.z[i]+=dt*k[2].z[i]; tmp.zdot[i]+=dt*k[2].zdot[i];}
    computeAccelerations(tmp,p,zddot_k);
    for(int i=0;i<9;++i){k[3].z[i]=tmp.zdot[i]; k[3].zdot[i]=zddot_k[i];}

    for(int i=0;i<9;++i){
        state.z[i]    +=(dt/6.0)*(k[0].z[i]    +2*k[1].z[i]    +2*k[2].z[i]    +k[3].z[i]);
        state.zdot[i] +=(dt/6.0)*(k[0].zdot[i] +2*k[1].zdot[i] +2*k[2].zdot[i] +k[3].zdot[i]);
    }
    computeAccelerations(state, p, state.zddot);
}

// ====================== CLOSURE FROM theta1 ======================
void closureFromTheta1(double th1, const SystemParams& p,
                       double& Rx1, double& Ry1, double& th2,
                       double& Rx2, double& Ry2,
                       double& Rx3, double& Ry3, double& th3) {
    double Bx=p.L1*std::cos(th1), By=p.L1*std::sin(th1);
    double dx=p.L4-Bx, dy=-By;
    double d=std::sqrt(dx*dx+dy*dy);

    if(d>(p.L2+p.L3)||d<std::fabs(p.L2-p.L3)){
        th2=th1; th3=th1;
        Rx1=(p.L1/2)*std::cos(th1); Ry1=(p.L1/2)*std::sin(th1);
        Rx2=Rx1; Ry2=Ry1;
        Rx3=p.L4+(p.L3/2)*std::cos(th1); Ry3=(p.L3/2)*std::sin(th1);
        return;
    }
    double ex=dx/d, ey=dy/d;
    double a=(p.L2*p.L2-p.L3*p.L3+d*d)/(2.0*d);
    double h2=p.L2*p.L2-a*a; if(h2<0)h2=0;
    double h=std::sqrt(h2);
    double Px=Bx+a*ex, Py=By+a*ey;
    double C1x=Px-h*ey, C1y=Py+h*ex;
    double C2x=Px+h*ey, C2y=Py-h*ex;
    double Cx,Cy;
    if(C1y>=C2y){Cx=C1x;Cy=C1y;}else{Cx=C2x;Cy=C2y;}
    th2=std::atan2(Cy-By,Cx-Bx);
    th3=std::atan2(Cy,Cx-p.L4);
    Rx1=(p.L1/2)*std::cos(th1); Ry1=(p.L1/2)*std::sin(th1);
    Rx2=0.5*(Bx+Cx);            Ry2=0.5*(By+Cy);
    Rx3=p.L4+(p.L3/2)*std::cos(th3); Ry3=(p.L3/2)*std::sin(th3);
}

// ====================== IKF — INITIALIZE (Chapter 3) ======================
void ikfInitialize(IKFState& ikf) {
    // x_hat = 0 initially (Eq.28)
    for(int i=0;i<3;++i) ikf.x_hat[i]=0.0;

    // P: initial error covariance (Eq.30)
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) ikf.P[i][j]=0.0;
    ikf.P[0][0] = 1e-6;
    ikf.P[1][1] = 1e-6;
    ikf.P[2][2] = 0.1;
    ikf.P[1][2] = 0.0005;
    ikf.P[2][1] = 0.0005;

    // Sigma_P: process noise (Eq.32)
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) ikf.Sigma_P[i][j]=0.0;
    ikf.Sigma_P[2][2] = 0.02;

    // Sigma_S: measurement noise covariance (Eq.40)
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) ikf.Sigma_S[i][j]=0.0;
    const double sigma_z    = 0.001;
    const double sigma_zdot = 0.005;
    const double sigma_zddot= 0.1;
    ikf.Sigma_S[0][0] = sigma_z     * sigma_z;
    ikf.Sigma_S[1][1] = sigma_zdot  * sigma_zdot;
    ikf.Sigma_S[2][2] = sigma_zddot * sigma_zddot;
}

// Compute acceleration Jacobians numerically for f_bar_x (Eq.31)
void computeAccelJacobians(const State& est, const SystemParams& p,
                            double& dZddot_dZ, double& dZddot_dZdot)
{
    const double eps = 1e-5;
    const double th1  = est.z[2];
    const double om1  = est.zdot[2];
    const double L1h  = p.L1/2.0;
    const double L2h  = p.L2/2.0;
    const double L3h  = p.L3/2.0;

    auto makeStateFromTheta = [&](double th1_p, double om1_v) -> State {
        State s;
        double Rx1,Ry1,th2,Rx2,Ry2,Rx3,Ry3,th3;
        closureFromTheta1(th1_p, p, Rx1,Ry1,th2,Rx2,Ry2,Rx3,Ry3,th3);
        s.z[0]=Rx1; s.z[1]=Ry1; s.z[2]=th1_p;
        s.z[3]=Rx2; s.z[4]=Ry2; s.z[5]=th2;
        s.z[6]=Rx3; s.z[7]=Ry3; s.z[8]=th3;

        double s1=std::sin(th1_p), c1=std::cos(th1_p);
        double s2=std::sin(th2),   c2=std::cos(th2);
        double s3=std::sin(th3),   c3=std::cos(th3);
        double a11=p.L2*c2, a12=-p.L3*c3;
        double a21=p.L2*s2, a22=-p.L3*s3;
        double det=a11*a22-a12*a21;
        double om2=0.0, om3=0.0;
        if(std::fabs(det)>1e-8){
            double r1=-p.L1*om1_v*c1, r2=-p.L1*om1_v*s1;
            om2=(r1*a22-r2*a12)/det;
            om3=(a11*r2-a21*r1)/det;
        }
        s.zdot[2]=om1_v; s.zdot[5]=om2; s.zdot[8]=om3;
        s.zdot[0]=-L1h*s1*om1_v;
        s.zdot[1]= L1h*c1*om1_v;
        s.zdot[3]=s.zdot[0]-L2h*s2*om2;
        s.zdot[4]=s.zdot[1]+L2h*c2*om2;
        s.zdot[6]=-L3h*s3*om3;
        s.zdot[7]= L3h*c3*om3;
        for(int i=0;i<9;++i) s.zddot[i]=0.0;
        return s;
    };

    // d(zddoti)/d(zi) — numerical central difference
    State sp = makeStateFromTheta(th1+eps, om1);
    State sm = makeStateFromTheta(th1-eps, om1);
    double zddot_p[9], zddot_m[9];
    computeAccelerations(sp, p, zddot_p);
    computeAccelerations(sm, p, zddot_m);
    dZddot_dZ = (zddot_p[2] - zddot_m[2]) / (2.0*eps);

    // d(zddoti)/d(zdoti) — numerical central difference
    State sqp = makeStateFromTheta(th1, om1+eps);
    State sqm = makeStateFromTheta(th1, om1-eps);
    double zddot_qp[9], zddot_qm[9];
    computeAccelerations(sqp, p, zddot_qp);
    computeAccelerations(sqm, p, zddot_qm);
    dZddot_dZdot = (zddot_qp[2] - zddot_qm[2]) / (2.0*eps);
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
    for(int i=0;i<3;++i) ikf.x_hat[i] = 0.0;
}

// ====================== MEASUREMENT MODEL (Eq.50) ======================
void measurementH(const State& s, double h_out[3]) {
    h_out[0] = s.z[2];       // theta
    h_out[1] = s.zdot[2];    // theta_dot
    h_out[2] = s.zddot[2];   // theta_ddot
}

// Analytical hx — Section 3.5.2 (Eq.49h)
void computeHx_analytical(const State& est, const SystemParams& p,
                           double hx[3][3]) {

    // Phi_z at current z (Eq.24a)
    double Phi_z[8][9];
    fourbarPhiz_numerical(est.z, p, Phi_z);

    // Partition Phi_z into Phi_iz and Phi_dz
    double Phi_iz[8], Phi_dz[8][8];
    partitionJacobian(Phi_z, Phi_iz, Phi_dz);

    // Solve Phi_dz * s = Phi_iz
    double s[8] = {0};
    gaussianElimination8(Phi_dz, Phi_iz, s);

    // Build dz/dzi and dzdot/dzdoti from constraint linearisation
    int dep_cols[8] = {0,1,3,4,5,6,7,8};
    double dz_dzi[9]       = {0};
    double dzdot_dzdoti[9] = {0};
    dz_dzi[2]       = 1.0;
    dzdot_dzdoti[2] = 1.0;
    for(int j = 0; j < 8; ++j){
        dz_dzi[dep_cols[j]]       = -s[j];
        dzdot_dzdoti[dep_cols[j]] = -s[j];
    }

    // Assemble hx (Eq.49h)
    hx[0][0] = dz_dzi[2];         // = 1
    hx[1][0] = 0.0;
    hx[2][0] = 0.0;

    hx[0][1] = 0.0;
    hx[1][1] = dzdot_dzdoti[2];   // = 1
    hx[2][1] = 0.0;

    hx[0][2] = 0.0;
    hx[1][2] = 0.0;
    hx[2][2] = 0.0;               // acceleration row is zero
}

// ====================== IKF CORRECTION (Section 3.3, Eq.33–37) ======================
void ikfCorrect(IKFState& ikf, const State& est,
                double o_theta, double o_theta_dot, double o_theta_ddot,
                const SystemParams& p) {

    // hx — analytical Jacobian (Section 3.5.2)
    double hx[3][3];
    computeHx_analytical(est, p, hx);

    // Innovation y = o - h(z_bar, zdot_bar, zddot_bar) (Eq.33)
    double h0[3];
    measurementH(est, h0);
    double y[3];
    y[0] = o_theta     - h0[0];
    y[1] = o_theta_dot  - h0[1];
    y[2] = o_theta_ddot - h0[2];

    // S = hx * P_minus * hx^T + Sigma_S (Eq.34)
    double hxP[3][3];
    for(int r=0; r<3; ++r)
        for(int c=0; c<3; ++c){
            hxP[r][c] = 0.0;
            for(int k=0; k<3; ++k)
                hxP[r][c] += hx[r][k] * ikf.P[k][c];
        }

    double S[3][3];
    for(int r=0; r<3; ++r)
        for(int c=0; c<3; ++c){
            S[r][c] = ikf.Sigma_S[r][c];
            for(int k=0; k<3; ++k)
                S[r][c] += hxP[r][k] * hx[c][k];
        }

    // S inverse
    double S_inv[3][3];
    if(!invert3x3(S, S_inv)) return;

    // K = P_minus * hx^T * S^{-1} (Eq.35)
    double PhxT[3][3];
    for(int r=0; r<3; ++r)
        for(int c=0; c<3; ++c){
            PhxT[r][c] = 0.0;
            for(int k=0; k<3; ++k)
                PhxT[r][c] += ikf.P[r][k] * hx[c][k];
        }

    double K[3][3];
    for(int r=0; r<3; ++r)
        for(int c=0; c<3; ++c){
            K[r][c] = 0.0;
            for(int k=0; k<3; ++k)
                K[r][c] += PhxT[r][k] * S_inv[k][c];
        }

    // x_hat_plus = 0 + K * y (Eq.36)
    for(int i=0; i<3; ++i){
        ikf.x_hat[i] = 0.0;
        for(int j=0; j<3; ++j)
            ikf.x_hat[i] += K[i][j] * y[j];
    }

    // P_plus = (I - K * hx) * P_minus (Eq.37)
    double I_KH[3][3];
    for(int r=0; r<3; ++r)
        for(int c=0; c<3; ++c){
            double kh = 0.0;
            for(int k=0; k<3; ++k)
                kh += K[r][k] * hx[k][c];
            I_KH[r][c] = (r==c ? 1.0 : 0.0) - kh;
        }

    double P_new[3][3];
    mult3x3(I_KH, ikf.P, P_new);
    for(int r=0; r<3; ++r)
        for(int c=0; c<3; ++c)
            ikf.P[r][c] = P_new[r][c];
}

// ====================== POST-CORRECTION (Section 3.4, Eq.41–48) ======================
void ikfApplyCorrection(State& est, const IKFState& ikf,
                        const SystemParams& p) {

    // IKF error state (Eq.28)
    const double delta_zi    = ikf.x_hat[0];
    const double delta_zdoti = ikf.x_hat[1];
    const double delta_zddoti= ikf.x_hat[2];

    // Corrected independent coordinates
    const double zi_hat    = est.z[2]    + delta_zi;
    const double zdoti_hat = est.zdot[2] + delta_zdoti;
    const double zddoti_hat= est.zddot[2]+ delta_zddoti;

    // POSITION CORRECTION (Eq.41–42)
    double Rx1,Ry1,th2,Rx2,Ry2,Rx3,Ry3,th3;
    closureFromTheta1(zi_hat, p, Rx1,Ry1,th2,Rx2,Ry2,Rx3,Ry3,th3);
    est.z[0]=Rx1; est.z[1]=Ry1; est.z[2]=zi_hat;
    est.z[3]=Rx2; est.z[4]=Ry2; est.z[5]=th2;
    est.z[6]=Rx3; est.z[7]=Ry3; est.z[8]=th3;

    // VELOCITY CORRECTION (Eq.43–45)
    const double s1=std::sin(zi_hat), c1=std::cos(zi_hat);
    const double s2=std::sin(th2),    c2=std::cos(th2);
    const double s3=std::sin(th3),    c3=std::cos(th3);

    double a11=p.L2*c2, a12=-p.L3*c3;
    double a21=p.L2*s2, a22=-p.L3*s3;
    double det=a11*a22-a12*a21;
    double om2=0.0, om3=0.0;
    if(std::fabs(det)>1e-8){
        double r1=-p.L1*zdoti_hat*c1, r2=-p.L1*zdoti_hat*s1;
        om2=(r1*a22-r2*a12)/det;
        om3=(a11*r2-a21*r1)/det;
    }
    const double L1h=p.L1/2.0, L2h=p.L2/2.0, L3h=p.L3/2.0;
    est.zdot[2]=zdoti_hat; est.zdot[5]=om2; est.zdot[8]=om3;
    est.zdot[0]=-L1h*s1*zdoti_hat;
    est.zdot[1]= L1h*c1*zdoti_hat;
    est.zdot[3]=est.zdot[0]-L2h*s2*om2;
    est.zdot[4]=est.zdot[1]+L2h*c2*om2;
    est.zdot[6]=-L3h*s3*om3;
    est.zdot[7]= L3h*c3*om3;

    // ACCELERATION CORRECTION (Eq.46–48)
    double Phi_z[8][9];
    fourbarPhiz_numerical(est.z, p, Phi_z);

    double Phi_iz[8], Phi_dz[8][8];
    partitionJacobian(Phi_z, Phi_iz, Phi_dz);

    // gamma — numerical (Eq.13b)
    const double eps=1e-6;
    double dPhiz_dz_zdot[8][9];
    for(int j=0;j<9;++j){
        double zp[9],zm[9];
        for(int k=0;k<9;++k){zp[k]=est.z[k];zm[k]=est.z[k];}
        zp[j]+=eps; zm[j]-=eps;
        double Phi_zp[8][9],Phi_zm[8][9];
        fourbarPhiz_numerical(zp,p,Phi_zp);
        fourbarPhiz_numerical(zm,p,Phi_zm);
        for(int i=0;i<8;++i){
            double cp=0.0,cm=0.0;
            for(int k=0;k<9;++k){cp+=Phi_zp[i][k]*est.zdot[k]; cm+=Phi_zm[i][k]*est.zdot[k];}
            dPhiz_dz_zdot[i][j]=(cp-cm)/(2.0*eps);
        }
    }
    double gamma[8];
    for(int i=0;i<8;++i){
        gamma[i]=0.0;
        for(int j=0;j<9;++j) gamma[i]-=dPhiz_dz_zdot[i][j]*est.zdot[j];
    }

    // RHS for dependent accelerations (Eq.47)
    double rhs_acc[8];
    for(int i=0;i<8;++i)
        rhs_acc[i] = gamma[i] - Phi_iz[i]*zddoti_hat;

    // Solve Phi_dz * zddot_d = rhs_acc (Eq.47)
    double zddot_dep[8];
    if(gaussianElimination8(Phi_dz, rhs_acc, zddot_dep)){
        int dep_cols[8]={0,1,3,4,5,6,7,8};
        for(int j=0;j<8;++j)
            est.zddot[dep_cols[j]] = zddot_dep[j];
    }
    // Independent acceleration (Eq.46)
    est.zddot[2] = zddoti_hat;
}

// ====================== CSV SAVE ======================
void saveToCSV() {
    std::ofstream csv("FourbarAL_IKF_Analytical.csv");
    csv << "Time,"
           "o_theta,o_theta_dot,o_theta_ddot,"
           "theta_cpp,theta_dot_cpp,theta_ddot_cpp,"
           "theta_ikf,theta_dot_ikf,theta_ddot_ikf,"
           "CycleTime_us,t_recv_us,t_ikf_us,t_save_us,t_other_us,"
           "P_zi,P_zdoti,P_zddoti\n";
    csv << std::fixed << std::setprecision(8);
    for(size_t i=0;i<logData.time.size();++i){
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
            << logData.t_recv_us[i]        << ","
            << logData.t_ikf_us[i]         << ","
            << logData.t_save_us[i]        << ","
            << logData.t_other_us[i]       << ","
            << logData.P_zi[i]             << ","
            << logData.P_zdoti[i]          << ","
            << logData.P_zddoti[i]         << "\n";
    }
    csv.close();
    std::cout<<"\nSaved "<<logData.time.size()
             <<" samples to FourbarAL_IKF_Analytical.csv\n";
}

// ====================== MAIN (Chapter 4) ======================
int main() {
    const double g =9.81;
    const double L1=2.0, L2=8.0, L3=5.0, L4=10.2;
    const double m1=2.0, m2=8.0, m3=5.0;
    auto rodI=[](double m,double L){return m*L*L/12.0;};  // Eq.25

    SystemParams p;
    p.g=g; p.L1=L1; p.L2=L2; p.L3=L3; p.L4=L4;
    p.m1=m1; p.m2=m2; p.m3=m3;
    p.I1=rodI(m1,L1); p.I2=rodI(m2,L2); p.I3=rodI(m3,L3);
    p.T0=0.0; p.win=2.0*PI*0.02;

    // TCP/IP server (Section 4.5)
    const int serverPort=5556;
    int sock=socket(AF_INET,SOCK_STREAM,0);
    if(sock<0){std::cerr<<"Socket creation failed\n";return 1;}
    int opt=1;
    setsockopt(sock,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));
    struct sockaddr_in addr;
    addr.sin_family=AF_INET;
    addr.sin_port=htons(serverPort);
    addr.sin_addr.s_addr=INADDR_ANY;
    if(bind(sock,(struct sockaddr*)&addr,sizeof(addr))<0){
        std::cerr<<"Bind failed\n";return 1;}
    listen(sock,1);
    std::cout<<"Waiting for MATLAB client on port "<<serverPort<<"...\n";
    int client=accept(sock,NULL,NULL);
    if(client<0){std::cerr<<"Accept failed\n";return 1;}
    std::cout<<"MATLAB connected!\n";

    // Initial conditions (Eq.15)
    const double th1_0=PI/6.0;
    double Rx1_0,Ry1_0,th2_0,Rx2_0,Ry2_0,Rx3_0,Ry3_0,th3_0;
    closureFromTheta1(th1_0,p,Rx1_0,Ry1_0,th2_0,Rx2_0,Ry2_0,Rx3_0,Ry3_0,th3_0);

    State plant, est;
    plant.z[0]=Rx1_0; plant.z[1]=Ry1_0; plant.z[2]=th1_0;
    plant.z[3]=Rx2_0; plant.z[4]=Ry2_0; plant.z[5]=th2_0;
    plant.z[6]=Rx3_0; plant.z[7]=Ry3_0; plant.z[8]=th3_0;
    for(int i=0;i<9;++i){plant.zdot[i]=0.0; plant.zddot[i]=0.0;}
    est=plant;

    // Compute initial accelerations (Eq.14b)
    computeAccelerations(plant, p, plant.zddot);
    computeAccelerations(est,   p, est.zddot);

    IKFState ikf;
    ikfInitialize(ikf);

    const double dt_tcp=0.001;  // 1 kHz (Section 4.1)
    const int    Nsamples=10000;
    int    samplecount=0;
    double max_cycle_time=0.0;

    while(samplecount<Nsamples){
        auto step_start=std::chrono::high_resolution_clock::now();

        // Receive theta, theta_dot, theta_ddot from MATLAB (Section 4.3.1)
        double o_theta_recv, o_theta_dot_recv, o_theta_ddot_recv;
        if(recvAll(client,&o_theta_recv,   sizeof(double))<=0 ||
           recvAll(client,&o_theta_dot_recv, sizeof(double))<=0 ||
           recvAll(client,&o_theta_ddot_recv,sizeof(double))<=0){
            std::cout<<"\nConnection lost.\n"; break;
        }
        o_theta_recv    =swapDouble(o_theta_recv);
        o_theta_dot_recv =swapDouble(o_theta_dot_recv);
        o_theta_ddot_recv=swapDouble(o_theta_ddot_recv);

        auto t_after_recv=std::chrono::high_resolution_clock::now();

        samplecount++;
        double t=samplecount*dt_tcp;

        // 1) Plant integration — comparison reference (Eq.52)
        rk4Step(plant, dt_tcp, p);
        const double theta_cpp_val     = plant.z[2];
        const double theta_dot_cpp_val = plant.zdot[2];
        const double theta_ddot_cpp_val= plant.zddot[2];

        // 2) MATLAB measurements o = [theta, theta_dot, theta_ddot]
        const double o_theta      = o_theta_recv;
        const double o_theta_dot  = o_theta_dot_recv;
        const double o_theta_ddot = o_theta_ddot_recv;

        // 3) IKF PREDICTION (Section 3.2, Eq.29–32)
        ikfPredict(ikf, est, p, dt_tcp);

        // 4) Propagate nominal trajectory to time k+1 (Eq.52)
        rk4Step(est, dt_tcp, p);

        // 5) IKF CORRECTION (Section 3.3, Eq.33–37)
        ikfCorrect(ikf, est, o_theta, o_theta_dot, o_theta_ddot, p);

        // 6) POST-CORRECTION (Section 3.4, Eq.41–48)
        ikfApplyCorrection(est, ikf, p);

        // 7) Extract IKF estimates
        const double theta_ikf_val     = est.z[2];
        const double theta_dot_ikf_val = est.zdot[2];
        const double theta_ddot_ikf_val= est.zddot[2];

        auto t_after_ikf=std::chrono::high_resolution_clock::now();

        // 8) Performance logging
        double error=o_theta-theta_ikf_val;

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
        // P diagonal elements (after correction — P_plus)
        logData.P_zi.push_back(ikf.P[0][0]);
        logData.P_zdoti.push_back(ikf.P[1][1]);
        logData.P_zddoti.push_back(ikf.P[2][2]);

        auto t_after_save=std::chrono::high_resolution_clock::now();

        // Sub-step timing (microseconds)
        double recv_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_after_recv - step_start).count();
        double ikf_us  = std::chrono::duration_cast<std::chrono::microseconds>(
            t_after_ikf  - t_after_recv).count();
        double save_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_after_save - t_after_ikf).count();
        double cycle_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_after_save - step_start).count();
        double other_us = cycle_time_us - recv_us - ikf_us - save_us;

        max_cycle_time=std::max(max_cycle_time,cycle_time_us);

        logData.cycle_time_us.push_back(cycle_time_us);
        logData.t_recv_us.push_back(recv_us);
        logData.t_ikf_us.push_back(ikf_us);
        logData.t_save_us.push_back(save_us);
        logData.t_other_us.push_back(other_us);

        if(samplecount%1000==0){
            printf("%8.3f %10.6f %10.6f %10.6f"
                   " %10.6f %10.6f %10.6f"
                   " %8.4f %8.4f %8.4f"
                   "  %8.5f  %10.8f  %8.2f us"
                   "  (recv=%.0f ikf=%.0f save=%.0f other=%.0f)\n",
                   t,
                   o_theta,         theta_cpp_val,     theta_ikf_val,
                   o_theta_dot,     theta_dot_cpp_val,  theta_dot_ikf_val,
                   o_theta_ddot,    theta_ddot_cpp_val, theta_ddot_ikf_val,
                   ikf.x_hat[2],
                   error, cycle_time_us,
                   recv_us, ikf_us, save_us, other_us);
        }
    }

    close(client);
    close(sock);
    saveToCSV();
    return 0;
}
