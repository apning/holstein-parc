//
//  main.cpp
//  Quench_Holstein
//
//  Created by Gia-Wei Chern on 11/22/20.
//  Modifications by Lingyu Yang
//  Additional minor modifications by Alex Ning mostly for adapting code to be a part of a specific data generation pipeline. Included removing some non-applicable parts of earlier code


#include <filesystem>
#include "util.hpp"
#include "model.hpp"

void simulation1(int argc, const char * argv[]) {

    std::filesystem::path directory = argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path("./");
    int sim_num                 = argc > 2  ? atoi(argv[2])  : 0;
    int phase                   = argc > 3  ? atoi(argv[3])  : 0;
    int save_interval           = argc > 4  ? atoi(argv[4])  : 8;
    int save_mid_interval       = argc > 5  ? atoi(argv[5])  : 0; // Evaluated as a boolean
    int L                       = argc > 6  ? atoi(argv[6])  : 32;
    double dlt_t                = argc > 7  ? atof(argv[7])  : 0.01;
    int pre_steps               = argc > 8  ? atoi(argv[8])  : 0;
    unsigned long saved_steps   = argc > 9  ? atoi(argv[9])  : 1024;
    double g_i                  = argc > 10 ? atof(argv[10]) : 0.5;
    double g_f                  = argc > 11 ? atof(argv[11]) : 0.8;
    double randomness_level     = argc > 12 ? atof(argv[12]) : 0.0001;
    int zero_displacement       = argc > 13 ? atoi(argv[13]) : 0; // Evaluated as a boolean
    int zero_momentum           = argc > 14 ? atoi(argv[14]) : 0; // Evaluated as a boolean
    double onsite_V_term        = argc > 15 ? atof(argv[15]) : 0.0;
    double eta_1                = argc > 16 ? atof(argv[16]) : 0.8;
    double init_Q               = argc > 17 ? atof(argv[17]) : 0.250104;
    int seed                    = argc > 18 ? atoi(argv[18]) : 0;
    // double W                    = argc > 19 ? atof(argv[19]) : 15.0;     // disorder strength (unused)
    double kT_init              = argc > 20 ? atof(argv[20]) : 0.00000001;
    
    
    
    seed += (unsigned)time(NULL)+getpid();
    init_srand();

    cout << "g_i = " << g_i << "\t g_f = " << g_f << endl;
    cout << "seed = " << seed << endl;
    cout << "dt = " << dlt_t << ", \t max_steps = " << save_interval*saved_steps+phase << endl;
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



    system.init_quenched_disorder(onsite_V_term);  //  disorder-free initial CDW state

    system.solve_self_consistent_Q(init_Q, zero_displacement, zero_momentum);
    // system.init_staggered_lattice(init_Q);
    // cout << "check init_Q:" << init_Q << endl;
    system.G = g_f;
    system.G1 = system.eta * g_f;

    // system.init_quenched_disorder(W);

    system.simulate_quench(saved_steps, pre_steps, dlt_t, save_interval, save_mid_interval, directory, sim_num, phase);


}


int main(int argc, const char * argv[]) {
    simulation1(argc, argv);

    return 0;
}
