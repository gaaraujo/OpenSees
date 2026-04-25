// Code written/implemented by:	Kristijan Kolozvari (kkolozvari@fullerton.edu)
//								California State University, Fullerton 
//								Kutay Orakcal
//								Bogazici University, Istanbul, Turkey
//								John Wallace
//								University of California, Los Angeles
//
// Created: 10/2015
//
// Description: This file contains the class implementation for uniaxialMaterial 
// SteelMPF, which represents the well-known uniaxial constitutive nonlinear 
// hysteretic material model for steel proposed by Menegotto and Pinto (1973), 
// and extended by Filippou et al. (1983) to include isotropic strain hardening 
// effects.
//
// References:
// 1) Menegotto, M., and Pinto, P.E. (1973). Method of analysis of cyclically 
// loaded RC plane frames including changes in geometry and non-elastic behavior 
// of elements under normal force and bending. Preliminary Report IABSE, vol 13. 
// 2) Filippou F.C., Popov, E.P., and Bertero, V.V. (1983). "Effects of Bond 
// Deterioration on Hysteretic Behavior of Reinforced Concrete Joints". Report 
// EERC 83-19, Earthquake Engineering Research Center, University of California, Berkeley.
//
// Source: /usr/local/cvs/OpenSees/SRC/material/uniaxial/SteelMPF.h
//
// Rev: 3


#ifndef SteelMPF_h
#define SteelMPF_h

#include <UniaxialMaterial.h>

class SteelMPF : public UniaxialMaterial
{
public:
	// Optional ultimate stress cap parameters, passed as a single aggregate.
	// Extend this struct (not the constructor signatures) when adding new cap params.
	struct UltParams {
		double sigultp;
		double sigultn;
		double Rult0;
	};

	// Typical constructor
	SteelMPF(int tag, double sigyieldp, double sigyieldn,
		double E0, double bp, double bn, double R0, double a1, double a2);

	// Constructor with strain hardening parameters a3, a4, a5 and a6
	SteelMPF(int tag, double sigyieldp, double sigyieldn,
		double E0, double bp, double bn, double R0, double a1, double a2,
		double a3, double a4, double a5, double a6);

	// Canonical constructor — pass ult=nullptr (or omit) for no ultimate stress cap
	SteelMPF(int tag, double sigyieldp, double sigyieldn,
		double E0, double bp, double bn, double R0, double a1, double a2,
		double a3, double a4, double a5, double a6,
		const UltParams* ult);

	// Blank constructor
	SteelMPF();

	// Destructor
	~SteelMPF();

	const char *getClassType(void) const { return "SteelMPF"; };

	int setTrialStrain(double strain, double strainRate = 0.0);
	int setTrial(double strain, double &stress, double &tangent, double strainRate = 0.0);
	double getStrain(void);
	double getStress(void);
	double getTangent(void);
	double getInitialTangent(void) { return E0; };

	int commitState(void);
	int revertToLastCommit(void);
	int revertToStart(void);

	UniaxialMaterial *getCopy(void);

	int sendSelf(int commitTag, Channel &theChannel);
	int recvSelf(int commitTag, Channel &theChannel,
		FEM_ObjectBroker &theBroker);

	void Print(OPS_Stream &s, int flag = 0);

protected:

private:

	// Material Properties
	double sigyieldp;	// yield stress in tension
	double sigyieldn;	// yield stress in compression
	double eyieldp;		//(calculated in constructor)
	double eyieldn;		//(calculated in constructor)
	double E0;			// Initial stiffness (Young's Modulus)
	double bp;			// Strain hardening ratio in tension
	double bn;			// Strain hardening ratio in compression
	double R0;			// Initial value of the curvature parameter R 
	double a1;			// Curvature degradation parameter 
	double aa1;			// Curvature degradation parameter 
	double a2;			// Curvature degradation parameter 
	double a3;			// Isotropic hardening parameter 
	double a4;			// Isotropic hardening parameter 
	double a5;			// Isotropic hardening parameter 
	double a6;			// Isotropic hardening parameter 
	double sigultp;		// ultimate stress cap in tension
	double sigultn;		// ultimate stress cap in compression
	double Rult0;		// Initial value of the ultimate cap curvature parameter
	bool hasSigUlt;		// True when an ultimate stress cap is active

	// TRIAL State Variables
	double def;
	double F;
	double stif;

	// CONVERGED State Variables
	double defold;
	double Fold;
	double stifold;

	// TRIAL History Variables
	int inc;

	double Rptwoprev;
	double Rntwoprev;

	int outp;
	int outn;

	double erp;
	double sigrp;

	double ern;
	double sigrn;

	double erpmaxmax;
	double ernmaxmax;

	double e0p;
	double sig0p;

	double e0n;
	double sig0n;

	double erntwoprev;
	double sigrntwoprev;
	double e0ntwoprev;
	double sig0ntwoprev;

	double erptwoprev;
	double sigrptwoprev;
	double e0ptwoprev;
	double sig0ptwoprev;

	double Rp;
	double Rn;

	int nloop;

	// CONVERGED History Variables
	int incold;

	double Rptwoprevold; 
	double Rntwoprevold; 

	int outpold;
	int outnold;

	double erpold;
	double sigrpold;

	double ernold;
	double sigrnold;

	double erpmaxmaxold;
	double ernmaxmaxold;

	double e0pold;
	double sig0pold;

	double e0nold;
	double sig0nold;

	double erntwoprevold;
	double sigrntwoprevold;
	double e0ntwoprevold;
	double sig0ntwoprevold;

	double erptwoprevold;
	double sigrptwoprevold;
	double e0ptwoprevold;
	double sig0ptwoprevold;

	double Rpold;
	double Rnold;

	int nloopold;

	// Calculates the trial state variables based on the trial strain
	void determineTrialState(double dStrain);

	// Evaluates MP stress and tangent, dispatching to the cap variant when hasSigUlt is true.
	void evalMpStressTangent(double e, double er, double sigr,
		double e0, double sig0, double sigult, double R, double b,
		double &sig, double &Et) const;

};

#endif