//
//  main.cpp
//  Quench_Holstein
//
//  Created by GW Chern on 11/22/20.
//  Modifications by Lingyu Yang
//  Additional non-substantial minor modification by Alex Ning mostly for data saving and disabling some behaviors
//

#include <filesystem>
#include "util.hpp"
#include "model.hpp"

void simulation1(int argc, const char * argv[]) {

    std::filesystem::path directory = argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path("./");
    int sim_num             = argc > 2 ? atoi(argv[2]) : 0;
    int L                   = argc > 3 ? atoi(argv[3]) : 32;
    int save_period         = argc > 4 ? atoi(argv[4]) : 8;
    double dlt_t            = argc > 5 ? atof(argv[5]) : 0.01;
    unsigned long max_steps = argc > 6 ? atoi(argv[6]) : 16000;
    double g_i              = argc > 7 ? atof(argv[7]) : 0.5;
    double g_f              = argc > 8 ? atof(argv[8]) : 0.8;
    int random_Q_P          = argc > 9 ? atoi(argv[9]) : 0; // Evaluated as a boolean
    double randomness_level = argc > 10 ? atof(argv[10]) : 0.0001;
    int zero_momentum       = argc > 11 ? atoi(argv[11]) : 0; // Evaluated as a boolean
    double eta_1            = argc > 12 ? atof(argv[12]) : 0.8;
    double init_Q           = argc > 13 ? atof(argv[13]) : 0.250104;
    int seed                = argc > 14 ? atoi(argv[14]) : 0;
    double W                = argc > 15 ? atof(argv[15]) : 15.0;     // disorder strength
    double kT_init          = argc > 16 ? atof(argv[16]) : 0.00000001;
    
    
    seed += (unsigned)time(NULL)+getpid();
    init_srand();

    cout << "g_i = " << g_i << "\t g_f = " << g_f << endl;
    cout << "seed = " << seed << endl;
    cout << "dt = " << dlt_t << ", \t max_steps = " << max_steps << endl;
    cout << "L = " << L << endl;


    Linear_Chain system(L, seed, randomness_level);

    system.t1 = -1;
    system.K0 = 0.3;
    system.K1 = 0.0;
    system.mass = 1.;
    system.G = g_i;
    system.eta = eta_1;

    system.kT = kT_init;
    system.filling = 0.5;



    // system.init_quenched_disorder(0.0001);  //  disorder-free initial CDW state

    system.solve_self_consistent_Q(init_Q, random_Q_P, zero_momentum);
    // system.init_staggered_lattice(init_Q);
    // cout << "check init_Q:" << init_Q << endl;
    system.G = g_f;
    system.G1 = system.eta * g_f;

    // system.init_quenched_disorder(W);

    system.simulate_quench(max_steps, dlt_t, save_period, directory, sim_num);


}


int main(int argc, const char * argv[]) {
    simulation1(argc, argv);

    return 0;
}
