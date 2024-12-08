//
//  model.cpp
//  Quench_Holstein
//
//  Created by GW Chern on 11/22/20.
//

#include <filesystem>
#include <cassert>
#include "model.hpp"


using namespace std;

// {s1, s2} components of pauli matrix vector,
// sigma1     sigma2     sigma3
//  0  1       0 -I       1  0
//  1  0       I  0       0 -1
const Vec3<cx_double> pauli[2][2] {
    {{0, 0, 1}, {1, -_I, 0}},
    {{1, _I, 0}, {0, 0, -1}}
};

// void Linear_Chain::init_quenched_disorder(double W) {
//     std::uniform_real_distribution<double> rd(-W, W);
//     for(int i=0; i<Ns; i++) onsite_V[i] = rd(rng);
// }

arma::sp_cx_mat Linear_Chain::build_Hamiltonian(arma::vec & disp) {
    arma::sp_cx_mat H(dim, dim);
    for(int i=0; i<Ns; i++) {
        H(i, mod(i+1, L)) = t1;
        H(i, mod(i-1, L)) = t1;
    }
    for(int i=0; i<Ns; i++) {
        H(i, i) = - G * disp(i);
        // H(i, i) = onsite_V[i] - G * disp(i);
    }
    return H;
}

void Linear_Chain::compute_fermi_level(arma::vec & eigE) {
    double x1 = eigE(0);
    double x2 = eigE(eigE.size()-1);

    int max_bisection = 500;
    double eps_bisection = 1.e-12;

    int iter = 0;
    while(iter < max_bisection || fabs(x2 - x1) > eps_bisection) {
        double xm = 0.5*(x1 + x2);
        double density = 0;
        for(int i=0; i<eigE.size(); i++) {
            density += fermi_density(eigE(i), kT, xm);
        }
        density /= ((double) dim);

        if(density <= filling) x1 = xm;
        else x2 = xm;

        iter++;
    }
    mu = 0.5*(x1 + x2);
}

arma::cx_mat Linear_Chain::compute_density_matrix(arma::sp_cx_mat & H) {

    arma::cx_mat Hm(H);
    arma::cx_mat D(dim, dim);

    arma::vec eigval(dim);
    arma::cx_mat eigvec;

    arma::eig_sym(eigval, eigvec, Hm);

    // ofstream fs("eig.dat");
    // for(int r=0; r<dim; r++) {
    //     fs << eigval(r) << endl;
    // }
    // fs.close();

    compute_fermi_level(eigval);

    arma::vec fd_factor(dim);
    for(int i=0; i<dim; i++) fd_factor(i) = fermi_density(eigval(i), kT, mu);

    D.zeros();
    for(int a=0; a<dim; a++)
        for(int b=a; b<dim; b++) {

            cx_double sum = 0;
            for(int m=0; m<dim; m++) {
                sum += fd_factor(m) * conj(eigvec(b, m)) * eigvec(a, m);
            }
            D(a, b) = sum;
            if(a != b) D(b, a) = conj(sum);
        }

    return D;
}

arma::cx_mat Linear_Chain::compute_on_site_density_matrix(arma::sp_cx_mat & H) {

    arma::cx_mat Hm(H);
    arma::cx_mat D(dim, dim);

    arma::vec eigval(dim);
    arma::cx_mat eigvec;

    arma::eig_sym(eigval, eigvec, Hm);

    compute_fermi_level(eigval);

    arma::vec fd_factor(dim);
    for(int i=0; i<dim; i++) fd_factor(i) = fermi_density(eigval(i), kT, mu);

    D.zeros();

    for(int i=0; i<Ns; i++) {
        cx_double sum = 0;
        for(int m=0; m<dim; m++) {
            sum += fd_factor(m) * conj(eigvec(i, m)) * eigvec(i, m);
        }
        D(i, i) = sum;
    }
    return D;
}

arma::vec Linear_Chain::compute_dX(arma::vec & Pm) {
    arma::vec dX(dim);
    for(int i=0; i<Ns; i++) {
        dX(i) = K0 * Pm(i);
    }
    return dX;
}

arma::vec Linear_Chain::compute_dP(arma::vec & Xm, arma::cx_mat & Dm) {
    arma::vec dP(dim);
    dP.zeros();
    for(int i=0; i<Ns; i++) {
        dP(i) = -K0 * Xm(i);
        // for(int k=0; k<N_nn1; k++) {
        //     dP(i) += -K1 * Xm(site[i].nn1[k]->idx);
        // }
        dP(i) += G1 * (real(Dm(i, i)) - filling);
    }
    return dP;
}

void Linear_Chain::integrate_EOM_RK4(double dt) {

    arma::sp_cx_mat H(Hamiltonian);
    arma::cx_mat D(Density_Mat);
    arma::cx_mat D2, KD, KD_sum;

    arma::vec X(displacement);
    arma::vec P(momentum);
    arma::vec X2, KX, KX_sum;
    arma::vec P2, KP, KP_sum;

    // ------- RK4 step-1: ----------------

    KD = -_I * dt * ( H * D - D * H );
    KX = dt * compute_dX(P);
    KP = dt * compute_dP(X, D);

    D2 = D + 0.5 * KD;
    KD_sum = KD / 6.;

    X2 = X + 0.5 * KX;
    KX_sum = KX / 6.;

    P2 = P + 0.5 * KP;
    KP_sum = KP / 6.;

    // ------- RK4 step-2: ----------------

    H = build_Hamiltonian(X2);
    KD = -_I * dt * ( H * D2 - D2 * H );

    KX = dt * compute_dX(P2);
    KP = dt * compute_dP(X2, D2);

    D2 = D + 0.5 * KD;
    KD_sum += KD / 3.;

    X2 = X + 0.5 * KX;
    KX_sum += KX / 3.;

    P2 = P + 0.5 * KP;
    KP_sum += KP / 3.;

    // ------- RK4 step-3: ----------------

    H = build_Hamiltonian(X2);
    KD = -_I * dt * ( H * D2 - D2 * H );

    KX = dt * compute_dX(P2);
    KP = dt * compute_dP(X2, D2);

    D2 = D + KD;
    KD_sum += KD / 3.;

    X2 = X + KX;
    KX_sum += KX / 3.;

    P2 = P + KP;
    KP_sum += KP / 3.;

    // ------- RK4 step-4: ----------------

    H = build_Hamiltonian(X2);
    KD = -_I * dt * ( H * D2 - D2 * H );
    KD_sum += KD / 6.;

    KX = dt * compute_dX(P2);
    KX_sum += KX / 6.;

    KP = dt * compute_dP(X2, D2);
    KP_sum += KP / 6.;

    // ------- RK4: sum all steps: ------------

    Density_Mat = D + KD_sum;

    displacement = X + KX_sum;
    momentum = P + KP_sum;

    // compute the system Hamiltonian, R, Delta:

    Hamiltonian = build_Hamiltonian(displacement);

}

// void Linear_Chain::init_random_displacements(double r) {
//     std::normal_distribution<double> rn;    // default mean = 0, var = 1

//     for(int i=0; i<Ns; i++) {
//         //displacement(i) = sqrt(2. * kT / K0) * rn(rng);
//         displacement(i) = r * rn(rng);
//     }
// }

// void Linear_Chain::init_random_momenta(double r) {
//     if(r == 0) {
//         for(int i=0; i<Ns; i++) momentum(i) = 0;
//     } else {
//         std::normal_distribution<double> rn;    // default mean = 0, var = 1

//         for(int i=0; i<Ns; i++) {
//             //momentum(i) = sqrt(2. * mass * kT) * rn(rng);
//             momentum(i) = r * rn(rng);
//         }
//     }
// }

void Linear_Chain::init_elastic_system(void) {
    std::normal_distribution<double> rn;    // default mean = 0, var = 1
    double r = randomness_level;
    for(int i=0; i<Ns; i++) {
        displacement(i) = r * rn(rng);
        momentum(i) = r * rn(rng);
    }
}

void Linear_Chain::compute_elastic_energy(void) {
    double sum1 = 0, sum2 = 0, sum3 = 0;
    for(int i=0; i<Ns; i++) {
        sum1 += pow(momentum(i), 2);
        sum2 += pow(displacement(i), 2);
        // for(int k=0; k<N_nn1; k++) {
        //     int j = site[i].nn1[k]->idx;
        //     sum3 += displacement(i) * displacement(j);
        // }
    }
    E_k = (K0 * sum1) / (2. * eta * (double) Ns);
    E_p = (K0 * sum2) / (2. * eta * (double) Ns);//+ 0.5 * K1 * sum3
}

void Linear_Chain::compute_electronic_energy(void) {
    cx_double sum = 0;
    cx_double tmp;
    int j;
    for(int i=0; i<Ns; i++) {
        tmp = Hamiltonian(i, i);
        sum += tmp * Density_Mat(i, i);
        j = mod(i+1, L);
        tmp = Hamiltonian(i, j);
        sum += tmp * Density_Mat(j, i);
        j = mod(i-1, L);
        tmp = Hamiltonian(i, j);
        sum += tmp * Density_Mat(j, i);
    }
    E_e = sum.real() / ((double) Ns);
}

void Linear_Chain::compute_charge_order(void) {
    double sum = 0;
    for(int i=0; i<Ns; i++) {
        sum += site[i].sgn * real(Density_Mat(i, i));
    }
    AF_charge = sum / ((double) Ns);
}

void Linear_Chain::compute_lattice_order(void) {
    double sum = 0;
    for(int i=0; i<Ns; i++) {
        sum += site[i].sgn * displacement(i);
    }
    AF_lattice = sum / ((double) Ns);
}

arma::vec Linear_Chain::staggered_lattice(double Qm) {
    arma::vec disp(dim);
    for(int i=0; i<Ns; i++) {
        disp(i) = site[i].sgn * Qm;
    }
    return disp;
}

// void Linear_Chain::save_configuration(string const filename) {
//     string full_path = filename;
//     std::ofstream fs;
//     fs.open(full_path.c_str(), ios::out);
//     fs.precision(18);
//     for(int i=0; i<Ns; i++) {

//         fs << site[i].x << '\t' << site[i].y << '\t';
//         fs << real(Density_Mat(i, i)) << '\t';
//         fs << displacement(i) << '\t';
//         fs << site[i].sgn * (real(Density_Mat(i, i)) - filling) << '\t';
//         fs << displacement(i) << '\t';//site[i].sgn *
//         fs << site[i].sgn * displacement(i) << '\t';
//         fs << momentum(i) << '\t';
//         fs << endl;
//     }
//     fs.close();
// }

// void Linear_Chain::save_density_mat(string const filename, arma::cx_mat D) {
//     string full_path = filename;
//     std::ofstream fs;
//     fs.open(full_path.c_str(), ios::out);
//     fs.precision(18);
//     for(int i=0; i<Ns; i++) {
//         for(int j=0; j<Ns; j++){
//           fs << site[i].x << '\t' << site[i].y << '\t' << site[j].x << '\t' << site[j].y << '\t';
//           fs << real(D(i, j)) << '\t';
//           fs << endl;
//         }
//     }
//     fs.close();
// }


// From one iteration of intergration save the density matrix (D), displacement (Q), and momentum (P)
void Linear_Chain::save_iteration(const std::filesystem::path& filename, const arma::cx_mat& D, const arma::vec& Q, const arma::vec& P) {
    std::ofstream fs;
    fs.open(filename, std::ios::app);
    fs.precision(18);

    // Write a separator before each new recording
    fs << "######" << std::endl;

    for (int i = 0; i < Ns; ++i) {
        for (int j = 0; j < Ns; ++j) {
            fs << std::real(D(i, j)) << '\t' << std::imag(D(i, j)) << std::endl;
        }
    }

    fs << "######" << std::endl;

    for(int i=0; i<Ns; i++) {
        fs << Q(i) << endl;
    }

    fs << "######" << std::endl;

    for(int i=0; i<Ns; i++) {
        fs << P(i) << endl;
    }


    fs.close();
}


// This method has been modified so that it intentionally does not necessarily find a self-consistent Q, I think - Alex
double EPS_err = 1.e-12;
void Linear_Chain::solve_self_consistent_Q(double Q_p, int random_Q_P, int zero_momentum) {
    arma::vec disp(dim);
    arma::vec disp_next(dim);
    arma::vec error_list(dim);
    double err = 10;
    int max_iter = 3000;
    int r = 0;
    disp = staggered_lattice(Q_p);
    while(r < max_iter && err > EPS_err) {//
        Hamiltonian = build_Hamiltonian(disp);
        Density_Mat = compute_density_matrix(Hamiltonian);
        for(int i1=0; i1<Ns; i1++){
            disp_next(i1) = eta * G * (real(Density_Mat(i1, i1))-0.5) / K0;
            error_list(i1) = abs(disp(i1) - disp_next(i1));
        }
        err = min(error_list);
        disp = disp_next;
        r++;
        cout << "in progress:" << r << '\t' << disp(0) << '\t' << disp_next(0) << endl;
    }
    cout << "self consistent:" << r << '\t' << disp(0) << '\t' << disp_next(0) << endl;
    displacement = disp;


    if (random_Q_P)
        init_elastic_system();

    if (zero_momentum) {
        // Set momentum to 0
        for(int i=0; i<Ns; i++) momentum(i) = 0;
    }

    Hamiltonian = build_Hamiltonian(displacement);
    Density_Mat = compute_density_matrix(Hamiltonian);
    
}





void Linear_Chain::simulate_quench(unsigned long max_steps, double dt, int save_period, std::filesystem::path directory, int sim_num) {

    Hamiltonian = build_Hamiltonian(displacement);

    // THE following few lines will generate a random integer in [0, save_period) in order to "offset" the save period with

    // Random number generator
    std::random_device rd_int;
    std::mt19937 gen(rd_int());
    std::uniform_int_distribution<> dis(0, save_period-1);

    // Generate a random number between 0 and save_period (exclusive) for save offset
    int save_offset = dis(gen);
    cout << "SAVE OFFSET: " << save_offset << endl;

    // Create save paths for density matrix and displacement/momentum files
    std::filesystem::path save_path = directory / std::filesystem::path("data_sim_" + std::to_string(sim_num) + ".dat");

    for(unsigned long i=0; i<max_steps; i++) {

        if ((i-save_offset) % save_period == 0) {
            save_iteration(save_path, Density_Mat, displacement, momentum);
        }

        integrate_EOM_RK4(dt);
        if (i % 1000 == 0) {
            cout << "time step:" << '\t' << i << endl;
        }
        


        // Used to be used. Not anymore:
        
        // time_real += dt;

        // if(i % n_save == 0) {
        //     double n_tot = real( arma::trace(Density_Mat) );
        //     compute_electronic_energy();
        //     compute_elastic_energy();
        //     compute_charge_order();

        //     fs << time_real << '\t';
        //     fs << AF_charge << '\t';
        //     fs << E_e << '\t';
        //     fs << n_tot << endl;
        //     fy << time_real << '\t' << E_p << '\t' << E_k << '\t' << E_e << '\t' << E_p+E_k+E_e << endl;
        // }

        // if(i % n_save_config == 0 && i<=6000) {
        //     cout << "i = " << i << endl;
        //     save_configuration("c" + std::to_string(nx) + ".dat");
        //     nx++;
        // }
        // if(i == 6000) {
        //     cout << "i = " << i << endl;
        //     save_density_mat("dm_" + std::to_string(i) + ".dat", Density_Mat);
        // }
    }


    if ((max_steps-save_offset) % save_period == 0) {
            save_iteration(save_path, Density_Mat, displacement, momentum);
        }

}
