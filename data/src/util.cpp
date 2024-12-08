//
//  util.cpp
//  pyro-kondo-chain
//
//  Created by Gia-Wei Chern on 5/6/16.
//  Copyright © 2016 Gia-Wei Chern. All rights reserved.
//

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "util.hpp"

void init_srand(void) {
    time_t seconds;
    time(&seconds);
    //srand((unsigned int) seconds);
    srand48(seconds);
}

int mod(int x, int m) {
    if (x>=0 && x<m)
        return x;
    else if (x<0)
        return m-1-mod(-1-x,m);
    else
        return x%m;
}

void select_sort(double *a, int _L) {
    int i, j, minat;
    double min;
    for(i=0; i<_L - 1; i++) {
        minat = i;
        min = a[i];
        for(j=i+1; j<_L; j++) {
            if(min > a[j]) {
                minat = j;
                min = a[j];
            }
        }
        double tmp = a[i];
        a[i] = a[minat];
        a[minat] = tmp;
    }
}

void select_sort(double *a, int _L, int *idx) {
    for(int m=0; m<_L; m++) idx[m] = m;
    
    int i, j, minat;
    double min;
    for(i=0; i<_L - 1; i++) {
        minat = i;
        min = a[i];
        for(j=i+1; j<_L; j++) {
            if(min > a[j]) {
                minat = j;
                min = a[j];
            }
        }
        double tmp = a[i];
        a[i] = a[minat];
        a[minat] = tmp;
        
        int itmp = idx[i];
        idx[i] = idx[minat];
        idx[minat] = itmp;    
        
    }    
}

double fermi_density(double x, double kT, double mu) {
    double alpha = (x-mu)/std::abs(kT);
    if (kT < 1e-15 || std::abs(alpha) > 20) {
        return (x < mu) ? 1.0 : 0.0;
    }
    else {
        return 1.0/(exp(alpha)+1.0);
    }
}

