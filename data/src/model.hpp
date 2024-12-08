//
//  model.hpp
//  Quench_Holstein
//
//  Created by GW Chern on 11/22/20.
//

#ifndef model_hpp
#define model_hpp

#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <random>

#include "vec3.hpp"
#include "util.hpp"

using namespace std;

typedef mt19937 RNG;

class Linear_Chain {
public:
    int L, Ns;
    int dim;

    double t1;
    double mass;
    double K0, K1;
    double G;
    double G1;
    double eta;

    double filling;
    double mu;
    double kT;

    static constexpr int N_nn1 = 2;

    double time_real;

    RNG rng;

    class Site {
    public:
        int idx;
        int x, y;

        int sgn;

        Site *nn1[N_nn1];

        Site(void) { };

    } *site;

    double *onsite_V;

    arma::sp_cx_mat Hamiltonian;
    arma::cx_mat Density_Mat;

    arma::vec displacement;
    arma::vec momentum;

    double randomness_level;

    Linear_Chain(int linear_size, int seed_0, double randomness_level_in) {

        L = linear_size;
        Ns = L;
        dim = Ns;

        onsite_V = new double[Ns];

        site = new Site[Ns];

        Hamiltonian = arma::sp_cx_mat(dim, dim);
        Density_Mat = arma::cx_mat(dim, dim);

        displacement = arma::vec(dim);
        momentum = arma::vec(dim);

        // init_lattice();

        time_real = 0;

        std::random_device seed;
        rng = RNG(seed() + seed_0);

        randomness_level = randomness_level_in;

    };
    void init_lattice(void);
    void init_quenched_disorder(double W);

    void init_random_momenta(double r);
    void init_random_displacements(double r);
    void init_elastic_system(void);
    // void staggered_lattice(double Qm);
    arma::vec staggered_lattice(double Qm);

    void read_init_config(void);


    arma::sp_cx_mat build_Hamiltonian(arma::vec & disp);

    void compute_fermi_level(arma::vec & eigE);
    arma::cx_mat compute_density_matrix(arma::sp_cx_mat &);

    arma::cx_mat compute_on_site_density_matrix(arma::sp_cx_mat &);

    arma::vec compute_dX(arma::vec & Pm);
    arma::vec compute_dP(arma::vec & Xm, arma::cx_mat & Dm);

    void integrate_EOM_RK4(double dt);

    void simulate_quench(int max_steps, double dt, double x_i, double x_f, double W);

    void simulate_pump_probe(int max_steps, double dt, double W);
    void solve_self_consistent_Q(double Q_p, int random_Q_P, int zero_momentum);

    void simulate_quench(unsigned long max_steps, double dt, int save_period, std::filesystem::path directory, int sim_num);

    // ========================================


    double E_p = 0, E_k = 0;
    double E_e = 0;

    void analyze_data(void);

    void compute_elastic_energy(void);
    void compute_electronic_energy(void);

    double AF_charge = 0;
    double AF_lattice = 0;

    void compute_charge_order(void);
    void compute_lattice_order(void);

    arma::cx_double minimization_charge_order_state(void);

    void save_configuration(string const filename);
    void save_density_mat(string const filename, arma::cx_mat D);

    void save_iteration(const std::filesystem::path& filename, const arma::cx_mat& D, const arma::vec& Q, const arma::vec& P);

    void test(void);

    double A_max, A_width, A_center, A_omega;
    double Vector_A_field(void);

};

#endif /* model_hpp */
