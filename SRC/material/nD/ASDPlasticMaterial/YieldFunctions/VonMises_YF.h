/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
**                                                                    **
**                                                                    **
** (C) Copyright 1999, The Regents of the University of California    **
** All Rights Reserved.                                               **
**                                                                    **
** Commercial use of this program without express permission of the   **
** University of California, Berkeley, is strictly prohibited.  See   **
** file 'COPYRIGHT'  in main directory for information on usage and   **
** redistribution,  and for a DISCLAIMER OF ALL WARRANTIES.           **
**                                                                    **
** Developed by:                                                      **
**   Frank McKenna (fmckenna@ce.berkeley.edu)                         **
**   Gregory L. Fenves (fenves@ce.berkeley.edu)                       **
**   Filip C. Filippou (filippou@ce.berkeley.edu)                     **
**                                                                    **
** ****************************************************************** */
                                                                        
// Original implementation: José Abell (UANDES), Massimo Petracca (ASDEA)
//
// ASDPlasticMaterial
//
// Fully general templated material class for plasticity modeling

#ifndef VonMises_YF_H
#define VonMises_YF_H

#include "../EvolvingVariable.h"
#include "../YieldFunctionBase.h"
#include "cmath"
#include <iostream>

// Defines indices i,j,k,l,m,n,p,q,r,s and the kronecker_delta.
#include "../ASDPlasticMaterialGlobals.h"
using namespace ASDPlasticMaterialGlobals;




template<class AlphaHardeningType, class KHardeningType>
class VonMises_YF : public YieldFunctionBase<VonMises_YF<AlphaHardeningType, KHardeningType>> // CRTP
{
public:

    typedef EvolvingVariable<VoigtVector, AlphaHardeningType> AlphaType;
    typedef EvolvingVariable<VoigtScalar, KHardeningType> KType;

    VonMises_YF( AlphaType &alpha_in, KType& k_in):
        YieldFunctionBase<VonMises_YF<AlphaHardeningType, KHardeningType>>::YieldFunctionBase(), // Note here that we need to fully-qualify the type of YieldFunctionBase, e.g. use scope resolution :: to tell compiler which instance of YieldFunctionBase will be used :/
                alpha_(alpha_in), k_(k_in) {}

    double operator()(const VoigtVector& sigma) const
    {
        double p = -(sigma.trace())/3;
        static VoigtVector s = sigma.deviator();

        const VoigtVector &alpha = alpha_.getVariableConstReference();
        const double &k = k_.getVariableConstReference();
        auto tmp = (s - alpha).dot(s - alpha);
        return sqrt( tmp ) - SQRT_2_over_3 * k ;  // This one assumes p positive in tension
    }

    const VoigtVector& df_dsigma_ij(const VoigtVector& sigma)
    {
        const VoigtVector &alpha = alpha_.getVariableConstReference();

        result = sigma.deviator() - alpha;

        double den = sqrt(result.dot(result));
        if (abs(den) > 100*ASDPlasticMaterialGlobals::MACHINE_EPSILON)
            result = result / den;

        return result;

    }

    double xi_star_h_star(const VoigtVector& depsilon, const VoigtVector& m, const VoigtVector& sigma)
    {
        double dbl_result = 0.0;

        // const VoigtVector &alpha = alpha_.getVariableConstReference();
        // // const double &k = k_.getVariableConstReference();

        // //Zero the stress deviator
        // s *= 0;

        // //Compute stress deviator (s) and mean pressure (p)
        // double p;
        // sigma.compute_deviatoric_tensor(s, p); // here p is positive if in tension, so flip the sign

        // //Denominator of the expression
        // double den = sqrt((s(i, j) - alpha(i, j)) * (s(i, j) - alpha(i, j)));

        // if (den == 0)
        // {
        //     return 0;
        // }

        // // This is for the hardening of k
        // constexpr double df_dk = -SQRT_2_over_3;
        // dbl_result +=  df_dk * k_.getDerivative(depsilon, m, sigma);

        // //This is for the hardening of alpha
        // dbl_result +=  -(s(i, j) - alpha(i, j)) / den * alpha_.getDerivative(depsilon, m, sigma)(i, j);

        return dbl_result;
    }
    bool hasCorner() const{
        return false;
    }
    bool in_Apex(VoigtVector const& TrialStress)
    {
        std::cout<<"von Mises yield surface does not have a corner. This function should never be callled!"<<std::endl;
        return false;
    }
    double get_k() const{
        return k_.getVariableConstReference();
    }
    VoigtVector const& get_alpha() const{
        return alpha_.getVariableConstReference();
    }
private:

    AlphaType &alpha_;
    KType &k_;
    static VoigtVector s; //Stress deviator
    static VoigtVector result; //For returning VoigtVector's

};

template <class AlphaHardeningType,  class KHardeningType>
VoigtVector VonMises_YF<AlphaHardeningType, KHardeningType>::s(3, 3, 0.0);
template <class AlphaHardeningType,  class KHardeningType>
VoigtVector VonMises_YF<AlphaHardeningType, KHardeningType>::result(3, 3, 0.0);


#endif