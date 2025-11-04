#include <Eigen/Dense>
#include <complex>

#include "PrecisionOfTypes.h"

using namespace Eigen;


VectorXc Ket0(int nQ){

    int dim = 1 << nQ;  // equivalent to 2^nQ
    VectorXc psi = VectorXc::Zero(dim); //All zeros vector
    
    psi[0] = Complex(Real(1.0), Real(0.0));
    
    return psi;

}

VectorXc Ket1(int nQ){

    int dim =1 <<nQ;
    VectorXc psi = VectorXc::Zero(dim);

    psi[dim-1] = Complex(Real(1.0), Real(0.0));
    return psi;
}

VectorXc plus_state(int d) {
    int dim = 1 << d; 
    VectorXc psi(dim);
    psi.setOnes();  

    psi /= std::sqrt(static_cast<Real>(dim));

    return psi;
}

VectorXc minus_state(int d) {
    int dim = 1 << d;
    VectorXc psi(dim);
    for (int i = 0; i < dim; ++i) {
        int wt = __builtin_popcount(i); 
        psi[i] = (wt % 2 == 0) ? 1.0 : -1.0;
    }
    psi /= std::sqrt(static_cast<Real>(dim));
    return psi;
}