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
// Source: /usr/local/cvs/OpenSees/SRC/material/uniaxial/SteelMPF.cpp
//
// Rev: 3


#include <SteelMPF.h>
#include <Vector.h>
#include <Matrix.h>
#include <Channel.h>
#include <Information.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <elementAPI.h>
#define OPS_Export 

namespace {

// Menegotto-Pinto along strain-stress chord.
inline void mpStressTangent(double e, double er, double sigr,
	double e0, double sig0, double R, double b, double *sig, double *Et)
{
	const double est = (e - er) / (e0 - er);
	const double eR = pow(est, R);
	const double onePluseR = 1.0 + eR;
	const double phi = pow(onePluseR, 1.0 / R);
	const double c = (1.0 - b) / phi;
	*sig = sigr + (sig0 - sigr) * (b * est + c * est);
	if (Et != nullptr)
		*Et = ((sig0 - sigr) / (e0 - er)) * (b + c * (1.0 - eR / onePluseR));
}

// Φ(ε*, R) = (1 + ε*^R)^(-1/R). Assumes ε* >= 0.
inline double mpScale(double est, double R)
{
	return pow(1.0 + pow(est, R), -1.0 / R);
}

// dΦ/dε* at ε* >= 0; equals 0 at ε*=0 to avoid 0^(R-1) blow-up for R<1.
inline double mpScaleDerivative(double est, double R)
{
	if (est == 0.0)
		return 0.0;
	const double onePlus = 1.0 + pow(est, R);
	return -pow(est, R - 1.0) * pow(onePlus, -1.0 / R - 1.0);
}

// MP stress/tangent with an ultimate-stress cap σ_u (signed; positive for tension, negative for compression).
// εu is the breakpoint where the b-asymptote crosses σ_u; ultimate-branch curvature uses fixed Rult0 (not R).
inline void mpStressTangentWithUlt(double e, double er, double sigr,
	double e0, double sig0, double sigult, double E0,
	double R, double Rult0, double b, double *sig, double *Et)
{
	const double eult = b == 0.0 ? e0 : e0 + (sigult - sig0) / (b * E0);
	const double est = (e - er) / (e0 - er);
	const double estult = (e - er) / (eult - er);
	const double scale = mpScale(est, R);
	const double scaleUlt = mpScale(estult, Rult0);

	*sig = sigr + (sig0 - sigr) * est * (b * scaleUlt + (1.0 - b) * scale);

	if (Et != nullptr) {
		const double dEst = 1.0 / (e0 - er);
		const double dEstUlt = 1.0 / (eult - er);
		const double dScale = mpScaleDerivative(est, R);
		const double dScaleUlt = mpScaleDerivative(estult, Rult0);
		const double dSigst = b * (dEst * scaleUlt + est * dScaleUlt * dEstUlt)
			+ (1.0 - b) * dEst * (scale + est * dScale);
		*Et = (sig0 - sigr) * dSigst;
	}
}

inline double checkGreaterThan(int tag, const char *name, double value, double limit)
{
	if (value > limit)
		return value;
	const double smallest = nextafter(limit, DBL_MAX);
	opserr << "WARNING SteelMPF tag " << tag << " - " << name
		<< " (" << value << ") must be > " << limit
		<< "; setting " << name << " = " << smallest << endln;
	return smallest;
}

inline double checkLowerThan(int tag, const char *name, double value, double limit)
{
	if (value < limit)
		return value;
	const double largest = nextafter(limit, -DBL_MAX);
	opserr << "WARNING SteelMPF tag " << tag << " - " << name
		<< " (" << value << ") must be < " << limit
		<< "; setting " << name << " = " << largest << endln;
	return largest;
}

inline double checkGreaterOrEqual(int tag, const char *name, double value, double limit)
{
	if (value >= limit)
		return value;
	opserr << "WARNING SteelMPF tag " << tag << " - " << name
		<< " (" << value << ") must be >= " << limit
		<< "; setting " << name << " = " << limit << endln;
	return limit;
}

inline double checkLowerOrEqual(int tag, const char *name, double value, double limit)
{
	if (value <= limit)
		return value;
	opserr << "WARNING SteelMPF tag " << tag << " - " << name
		<< " (" << value << ") must be <= " << limit
		<< "; setting " << name << " = " << limit << endln;
	return limit;
}

} // namespace

void SteelMPF::evalMpStressTangent(double e, double er, double sigr,
	double e0, double sig0, double sigult, double R, double b,
	double &sig, double &Et) const
{
	if (hasSigUlt) {
		mpStressTangentWithUlt(e, er, sigr, e0, sig0, sigult, E0, R, Rult0, b, &sig, &Et);
		return;
	}

	mpStressTangent(e, er, sigr, e0, sig0, R, b, &sig, &Et);
}

// Read input parameters and build the material
OPS_Export void *OPS_SteelMPF(void)
{
	// Pointer to a uniaxial material that will be returned                       
	UniaxialMaterial *theMaterial = 0;

	const int totalNumArgs = OPS_GetNumRemainingInputArgs();

	if (totalNumArgs < 6) {
		opserr << "Incorrect # args, Want: uniaxialMaterial SteelMPF tag? sigyieldp? sigyieldn? "
			<< "E0? bp? bn? <R0? cR1? cR2? <a1? a2? a3? a4?>> <-ult sigultp? sigultn? Rult0?>" << endln;
		return 0;
	}

	// Scan trailing args for the optional -ult flag, which must be the last option.
	bool hasUlt = false;
	double ultData[3] = {0.0, 0.0, 0.0};
	int numUltArgs = 0;
	for (int scanned = 0; scanned < totalNumArgs; ++scanned) {
		const char *opt = OPS_GetString();
		if (opt != nullptr && strcmp(opt, "-ult") == 0) {
			if (OPS_GetNumRemainingInputArgs() != 3) {
				opserr << "WARNING SteelMPF -ult must be the last option, "
					<< "followed by sigultp? sigultn? Rult0?" << endln;
				return 0;
			}
			int n = 3;
			if (OPS_GetDoubleInput(&n, ultData) != 0) {
				opserr << "WARNING SteelMPF invalid -ult parameters" << endln;
				return 0;
			}
			hasUlt = true;
			numUltArgs = 4;	// flag + 3 doubles
			break;
		}
	}

	OPS_ResetCurrentInputArg(-totalNumArgs);

	const int numMPFArgs = totalNumArgs - numUltArgs;
	if (numMPFArgs != 6 && numMPFArgs != 9 && numMPFArgs != 13) {
		opserr << "Incorrect # mandatory args (" << numMPFArgs << "), Want: "
			<< "uniaxialMaterial SteelMPF tag? sigyieldp? sigyieldn? E0? bp? bn? "
			<< "<R0? cR1? cR2? <a1? a2? a3? a4?>> <-ult sigultp? sigultn? Rult0?>" << endln;
		return 0;
	}

	int iData[1];
	double dData[12];
	dData[5] = 20.0;	// R0 default
	dData[6] = 0.915;	// cR1 default
	dData[7] = 0.15;	// cR2 default
	dData[8] = 0.0;		// a3 default
	dData[9] = 1.0;		// a4 default
	dData[10] = 0.0;	// a5 default
	dData[11] = 1.0;	// a6 default

	int numData = 1;
	if (OPS_GetIntInput(&numData, iData) != 0) {
		opserr << "WARNING invalid uniaxialMaterial SteelMPF tag" << endln;
		return 0;
	}

	numData = numMPFArgs - 1;
	if (OPS_GetDoubleInput(&numData, dData) != 0) {
		opserr << "Invalid data for uniaxialMaterial SteelMPF " << dData[0] << endln;
		return 0;
	}

	SteelMPF::UltParams ult{ultData[0], ultData[1], ultData[2]};
	theMaterial = new SteelMPF(iData[0], dData[0], dData[1],
		dData[2], dData[3], dData[4], dData[5], dData[6], dData[7],
		dData[8], dData[9], dData[10], dData[11],
		hasUlt ? &ult : nullptr);

	return theMaterial;
}

// Default Constructor
SteelMPF::SteelMPF
	(int tag, double FYp, double FYn, double E, double Bp, double Bn, double Rr, double A1, double A2)
	: SteelMPF(tag, FYp, FYn, E, Bp, Bn, Rr, A1, A2, 0.0, 1.0, 0.0, 1.0)
{
}

// Constructor with optional strain hardening parameters a3, a4, a5 and a6
SteelMPF::SteelMPF
	(int tag, double FYp, double FYn, double E, double Bp, double Bn, double Rr, double A1, double A2, double A3, double A4, double A5, double A6)
	: SteelMPF(tag, FYp, FYn, E, Bp, Bn, Rr, A1, A2, A3, A4, A5, A6, nullptr)
{
}

// Canonical constructor — all parameters, owns all initialization
SteelMPF::SteelMPF
	(int tag, double FYp, double FYn, double E, double Bp, double Bn,
		double Rr, double A1, double A2, double A3, double A4, double A5, double A6,
		const UltParams* ult) :
UniaxialMaterial(tag, MAT_TAG_SteelMPF),
	sigyieldp(FYp), sigyieldn(FYn), E0(E), bp(Bp), bn(Bn), R0(Rr), aa1(A1), a2(A2), a3(A3), a4(A4), a5(A5), a6(A6),
	sigultp(DBL_MAX), sigultn(DBL_MAX), Rult0(Rr), hasSigUlt(ult != nullptr)

{
	R0  = checkGreaterThan(tag, "R0",  R0,  0.0);
	aa1 = checkGreaterOrEqual(tag, "cR1", aa1, 0.0);
	aa1 = checkLowerThan(tag, "cR1", aa1, 1.0);
	a2  = checkGreaterThan(tag, "cR2", a2,  0.0);

	if (ult != nullptr) {
		sigultp = checkGreaterOrEqual(tag, "sigultp", ult->sigultp, sigyieldp);
		sigultn = checkGreaterOrEqual(tag, "sigultn", ult->sigultn, sigyieldn);
		Rult0 = checkGreaterThan(tag, "Rult0", ult->Rult0, 0.0);
	}

	// Sets all history and state variables to initial values

	// TRIAL History variables
	inc = 0;

	Rptwoprev = 0.0;
	Rntwoprev = 0.0;

	outp = 0;
	outn = 0;

	erp = 0.0;
	sigrp = 0.0;

	ern = 0.0;
	sigrn = 0.0;

	erpmaxmax = 0.0;
	ernmaxmax = 0.0;

	e0p = 0.0;
	sig0p = 0.0;

	e0n = 0.0;
	sig0n = 0.0;

	erntwoprev = 0.0;
	sigrntwoprev = 0.0;
	e0ntwoprev = 0.0;
	sig0ntwoprev = 0.0;

	erptwoprev = 0.0;
	sigrptwoprev = 0.0;
	e0ptwoprev = 0.0;
	sig0ptwoprev = 0.0;

	Rp = 0.0;
	Rn = 0.0;

	nloop = 0;

	// CONVERGED History variables
	incold = 0;

	Rptwoprevold = 0.0;
	Rntwoprevold = 0.0;

	outpold = 0;
	outnold = 0;

	erpold = 0.0;
	sigrpold = 0.0;

	ernold = 0.0;
	sigrnold = 0.0;

	erpmaxmaxold = 0.0;
	ernmaxmaxold = 0.0;

	e0pold = 0.0;
	sig0pold = 0.0;

	e0nold = 0.0;
	sig0nold = 0.0;

	erntwoprevold = 0.0;
	sigrntwoprevold = 0.0;
	e0ntwoprevold = 0.0;
	sig0ntwoprevold = 0.0;

	erptwoprevold = 0.0;
	sigrptwoprevold = 0.0;
	e0ptwoprevold = 0.0;
	sig0ptwoprevold = 0.0;

	Rpold = R0;
	Rnold = R0;

	nloopold = 0;

	a1 = aa1*R0;

	// TRIAL State variables
	def = 0.0;
	F = 0.0;
	stif = E0;

	// CONVERGED State variables
	defold = 0.0;
	Fold = 0.0;
	stifold = E0;

	// Yield Strain
	eyieldp = sigyieldp/E0;
	eyieldn = sigyieldn/E0;

}

// blank constructor
SteelMPF::SteelMPF() :UniaxialMaterial(0, MAT_TAG_SteelMPF),
	sigyieldp(0.0), sigyieldn(0.0), E0(0.0), bp(0.0), bn(0.0), R0(0.0), aa1(0.0), a2(0.0), a3(0.0), a4(0.0), a5(0.0), a6(0.0), sigultp(DBL_MAX), sigultn(DBL_MAX), Rult0(0.0), hasSigUlt(false)
{

}

// Destructor
SteelMPF::~SteelMPF ()
{

}

int SteelMPF::setTrialStrain(double strain, double strainRate)
{
	// Trial strain; trial history is set from committed (*old) state inside determineTrialState
	def = strain;
	determineTrialState(def);

	return 0;
}


int SteelMPF::setTrial (double strain, double &stress, double &tangent, double strainRate)
{
	def = strain;
	determineTrialState(def);

	stress = F; 
	tangent = stif;

	return 0;
}

// Calculate trial stress and stiffness
void SteelMPF::determineTrialState(double def)
{

	double e = def;
	double eold = defold;
	double sigold = Fold;

	// Determine initial loading condition
	if (incold == 0) {

		static const double strainTol = 1.0e-14;

		Rptwoprev = R0;
		Rntwoprev = R0;
		outp = 1;
		outn = 1;

		if (e < -strainTol) {
			inc = -1;
		} else if (e > strainTol) {
			inc = 1;
		} else {
			inc = 0;
		}

		erp=0.0;
		sigrp=0.0;
		ern=0.0;
		sigrn=0.0;

		erptwoprev=0.0;
		sigrptwoprev=0.0;
		erntwoprev=0.0;
		sigrntwoprev=0.0;

		erpmaxmax=0.0;
		ernmaxmax=0.0;

		sig0p = sigyieldp;
		e0p = sig0p / E0;
		e0ptwoprev=e0p;
		sig0ptwoprev=sig0p;

		sig0n = -sigyieldn;
		e0n = sig0n / E0;
		e0ntwoprev=e0n;
		sig0ntwoprev=sig0n;

		Rp = R0;
		Rn = R0;

		if (inc == 0) {

			F = 0.0;
			stif = E0;
			nloop = 0;

		} else {

			nloop = 1;

			if (inc==1) {

				sig0p = sigyieldp;
				e0p = sig0p / E0;
				e0ptwoprev=e0p;
				sig0ptwoprev=sig0p;

				double sig, Et;
				evalMpStressTangent(e, erp, sigrp, e0p, sig0p, sigultp, Rp, bp, sig, Et);

				F = sig;
				stif = Et;

			} else {

				sig0n = -sigyieldn;
				e0n = sig0n / E0;
				e0ntwoprev=e0n;
				sig0ntwoprev=sig0n;

				double sig, Et;
				evalMpStressTangent(e, ern, sigrn, e0n, sig0n, -sigultn, Rn, bn, sig, Et);

				F = sig;
				stif = Et;

			}

		}

	}

	else {

		// Cyclic Stress - Strain model for Steel derived from Menegotto - Pinto Equation

		outp=outpold;
		outn=outnold;
		Rptwoprev=Rptwoprevold;
		Rntwoprev=Rntwoprevold;

		if (def > defold) {
			inc = 1;
		} else if (def < defold) {
			inc = -1;
		} else {
			inc = incold;
		}

		erp=erpold;
		sigrp=sigrpold;

		ern=ernold;
		sigrn=sigrnold;

		erpmaxmax=erpmaxmaxold;
		ernmaxmax=ernmaxmaxold;

		e0p=e0pold;
		sig0p=sig0pold;

		e0n=e0nold;
		sig0n=sig0nold;

		erntwoprev=erntwoprevold;
		sigrntwoprev=sigrntwoprevold;
		e0ntwoprev=e0ntwoprevold;
		sig0ntwoprev=sig0ntwoprevold;

		erptwoprev=erptwoprevold;
		sigrptwoprev=sigrptwoprevold;
		e0ptwoprev=e0ptwoprevold;
		sig0ptwoprev=sig0ptwoprevold;

		Rp=Rpold;
		Rn=Rnold;

		nloop=nloopold;

		if (incold == 1) {

			if (e < eold) {

				Rntwoprev=Rnold;    
				erntwoprev=ernold;
				sigrntwoprev=sigrnold;
				e0ntwoprev=e0nold;
				sig0ntwoprev=sig0nold;

				ern=eold;
				sigrn=sigold;
				nloop=nloopold+1;

				double sigy = -sigyieldn;

				if (ern > ernmaxmaxold) {
					ernmaxmax=ern;
				} else {
					ernmaxmax=ernmaxmaxold;
				}

				double sigstar=-sigy*a3*(abs(ernmaxmax)/eyieldp-a4);

				if (abs(ernmaxmax) < eyieldp) {
					sigstar = 0.0;
				}

				if (sigstar < 0.0) {
					sigstar = 0.0;
				}

				const double e0nbase = (sigy * (1.0 - bn) + E0 * ern - sigrn) / (E0 * (1.0 - bn));
				const double sig0nbase = sigrn + E0 * (e0nbase - ern);
				sig0n = sig0nbase - sigstar;
				if (hasSigUlt && sig0n < -sigultn) {
					sig0n = -sigultn;
				}
				e0n = ern + (sig0n - sigrn) / E0;

				double em;

				if (nloop==1) {
					em=e0n;
				} else if (nloop==2) {
					em=-eyieldn;
				} else {
					em=erp;
				}

				double ksi=abs((em-e0n)/eyieldn);

				Rn = R0 - a1*ksi/(a2+ksi);

				if (Rn > Rnold) {
					Rn = Rnold;
				}

				double sig, Et;
				evalMpStressTangent(e, ern, sigrn, e0n, sig0n, -sigultn, Rn, bn, sig, Et);

				double siglimit, Etlimit;
				evalMpStressTangent(e, erntwoprev, sigrntwoprev, e0ntwoprev,
					sig0ntwoprev, -sigultn, Rntwoprev, bn, siglimit, Etlimit);

				double sigcontrol, Etcontrol;
				evalMpStressTangent(ern, erntwoprev, sigrntwoprev, e0ntwoprev,
					sig0ntwoprev, -sigultn, Rntwoprev, bn, sigcontrol, Etcontrol);

				if (sigrn < sigcontrol) {
					outn=1;
				} else {
					outn=0;
				}

				if (ern < erntwoprev && outn==0) {

					if (sig < siglimit) {

						sig=siglimit;
						Et=Etlimit;
						ern=erntwoprev;
						sigrn=sigrntwoprev;
						e0n=e0ntwoprev;
						sig0n=sig0ntwoprev;
						Rn=Rntwoprev;

					}

				}

				F = sig;
				stif = Et;

			} else {

				Rp=Rpold;

				double sig, Et;
				evalMpStressTangent(e, erp, sigrp, e0p, sig0p, sigultp, Rp, bp, sig, Et);

				double siglimit, Etlimit;
				evalMpStressTangent(e, erptwoprev, sigrptwoprev, e0ptwoprev,
					sig0ptwoprev, sigultp, Rptwoprev, bp, siglimit, Etlimit);

				if (erp>erptwoprev && outp==0) {

					if (sig>siglimit) {

						sig=siglimit;
						Et=Etlimit;
						erp=erptwoprev;
						sigrp=sigrptwoprev;
						e0p=e0ptwoprev;
						sig0p=sig0ptwoprev;
						Rp=Rptwoprev;

					}

				}

				F = sig;
				stif = Et;

			}
		}

		else if (incold == -1) {

			if (e > eold) {

				Rptwoprev=Rpold;
				erptwoprev=erpold;
				sigrptwoprev=sigrpold;
				e0ptwoprev=e0pold;
				sig0ptwoprev=sig0pold;

				erp=eold;
				sigrp=sigold;
				nloop=nloopold+1;
				double sigy = sigyieldp;

				if (erp < erpmaxmaxold) {
					erpmaxmax=erp;
				} else {
					erpmaxmax=erpmaxmaxold;
				}

				double sigstar=sigy*a5*(abs(erpmaxmax)/eyieldn-a6);

				if (abs(erpmaxmax) < eyieldn) {
					sigstar = 0.0;
				}

				if (sigstar < 0.0) {
					sigstar = 0.0;
				}

				const double e0pbase = (sigy * (1.0 - bp) + E0 * erp - sigrp) / (E0 * (1.0 - bp));
				const double sig0pbase = sigrp + E0 * (e0pbase - erp);
				sig0p = sig0pbase + sigstar;
				if (hasSigUlt && sig0p > sigultp) {
					sig0p = sigultp;
				}
				e0p = erp + (sig0p - sigrp) / E0;

				double em;

				if (nloop==1) {
					em=e0p;
				} else if (nloop==2) {
					em=eyieldp;
				} else {
					em=ern;
				}

				double ksi = abs((em-e0p)/eyieldp);

				Rp = R0 - a1*ksi/(a2+ksi);

				if (Rp > Rpold) {
					Rp = Rpold;
				}

				double sig, Et;
				evalMpStressTangent(e, erp, sigrp, e0p, sig0p, sigultp, Rp, bp, sig, Et);

				double siglimit, Etlimit;
				evalMpStressTangent(e, erptwoprev, sigrptwoprev, e0ptwoprev,
					sig0ptwoprev, sigultp, Rptwoprev, bp, siglimit, Etlimit);

				double sigcontrol, Etcontrol;
				evalMpStressTangent(erp, erptwoprev, sigrptwoprev, e0ptwoprev,
					sig0ptwoprev, sigultp, Rptwoprev, bp, sigcontrol, Etcontrol);

				if (sigrp > sigcontrol) {
					outp=1;
				} else {
					outp=0;
				}

				if (erp > erptwoprev && outp == 0) {
					if (sig>siglimit){
						sig=siglimit;
						Et=Etlimit;
						erp=erptwoprev;
						sigrp=sigrptwoprev;
						e0p=e0ptwoprev;
						sig0p=sig0ptwoprev;
						Rp=Rptwoprev;
					}
				}

				F = sig;
				stif = Et;

			}

			else {

				Rn=Rnold;

				double sig, Et;
				evalMpStressTangent(e, ern, sigrn, e0n, sig0n, -sigultn, Rn, bn, sig, Et);

				double siglimit, Etlimit;
				evalMpStressTangent(e, erntwoprev, sigrntwoprev, e0ntwoprev,
					sig0ntwoprev, -sigultn, Rntwoprev, bn, siglimit, Etlimit);

				if (ern < erntwoprev && outn == 0) {

					if (sig < siglimit) {

						sig=siglimit;
						Et=Etlimit;
						ern=erntwoprev;
						sigrn=sigrntwoprev;
						e0n=e0ntwoprev;
						sig0n=sig0ntwoprev;
						Rn=Rntwoprev;

					}

				}

				F = sig;
				stif = Et;

			}

		}

	}

}

double SteelMPF::getStrain()
{
	return def;
}

double SteelMPF::getStress()
{
	return F;
}

double SteelMPF::getTangent()
{
	return stif;
}

int SteelMPF::commitState()
{

	// History variables
	incold = inc;

	Rptwoprevold = Rptwoprev;
	Rntwoprevold = Rntwoprev;

	outpold = outp;
	outnold = outn;

	erpold = erp;
	sigrpold = sigrp;

	ernold = ern;
	sigrnold = sigrn;

	erpmaxmaxold = erpmaxmax;
	ernmaxmaxold = ernmaxmax;

	e0pold = e0p;
	sig0pold = sig0p;

	e0nold = e0n;
	sig0nold = sig0n;

	erntwoprevold = erntwoprev;
	sigrntwoprevold = sigrntwoprev;
	e0ntwoprevold = e0ntwoprev;
	sig0ntwoprevold = sig0ntwoprev;

	erptwoprevold = erptwoprev;
	sigrptwoprevold = sigrptwoprev;
	e0ptwoprevold = e0ptwoprev;
	sig0ptwoprevold = sig0ptwoprev;

	Rpold = Rp;
	Rnold = Rn;

	nloopold = nloop;

	// State variables
	defold = def;
	Fold = F;
	stifold = stif;

	return 0;
}

int SteelMPF::revertToLastCommit()
{
	// Reset trial history variables to last committed state
	inc = incold;

	Rptwoprev = Rptwoprevold;
	Rntwoprev = Rntwoprevold;

	outp = outpold;
	outn = outnold;

	erp = erpold;
	sigrp = sigrpold;

	ern = ernold;
	sigrn = sigrnold;

	erpmaxmax = erpmaxmaxold;
	ernmaxmax = ernmaxmaxold;

	e0p = e0pold;
	sig0p = sig0pold;

	e0n = e0nold;
	sig0n = sig0nold;

	erntwoprev = erntwoprevold;
	sigrntwoprev = sigrntwoprevold;
	e0ntwoprev = e0ntwoprevold;
	sig0ntwoprev = sig0ntwoprevold;

	erptwoprev = erptwoprevold;
	sigrptwoprev = sigrptwoprevold;
	e0ptwoprev = e0ptwoprevold;
	sig0ptwoprev = sig0ptwoprevold;

	Rp = Rpold;
	Rn = Rnold;

	nloop = nloopold;

	// Reset trial state variables to last committed state
	def = defold;
	F = Fold;
	stif = stifold;

	return 0;
}

int SteelMPF::revertToStart()
{
	// TRIAL History variables
	inc = 0;

	Rptwoprev = 0.0;
	Rntwoprev = 0.0;

	outp = 0;
	outn = 0;

	erp = 0.0;
	sigrp = 0.0;

	ern = 0.0;
	sigrn = 0.0;

	erpmaxmax = 0.0;
	ernmaxmax = 0.0;

	e0p = 0.0;
	sig0p = 0.0;

	e0n = 0.0;
	sig0n = 0.0;

	erntwoprev = 0.0;
	sigrntwoprev = 0.0;
	e0ntwoprev = 0.0;
	sig0ntwoprev = 0.0;

	erptwoprev = 0.0;
	sigrptwoprev = 0.0;
	e0ptwoprev = 0.0;
	sig0ptwoprev = 0.0;

	Rp = R0;
	Rn = R0;

	nloop = 0;

	// CONVERGED History variables
	incold = 0;

	Rptwoprevold = 0.0;
	Rntwoprevold = 0.0;

	outpold = 0;
	outnold = 0;

	erpold = 0.0;
	sigrpold = 0.0;

	ernold = 0.0;
	sigrnold = 0.0;

	erpmaxmaxold = 0.0;
	ernmaxmaxold = 0.0;

	e0pold = 0.0;
	sig0pold = 0.0;

	e0nold = 0.0;
	sig0nold = 0.0;

	erntwoprevold = 0.0;
	sigrntwoprevold = 0.0;
	e0ntwoprevold = 0.0;
	sig0ntwoprevold = 0.0;

	erptwoprevold = 0.0;
	sigrptwoprevold = 0.0;
	e0ptwoprevold = 0.0;
	sig0ptwoprevold = 0.0;

	Rpold = R0;
	Rnold = R0;

	nloopold = 0;

	// TRIAL State variables
	def = 0.0;
	F = 0.0;
	stif = E0;

	// CONVERGED State variables
	defold = 0.0;
	Fold = 0.0;
	stifold = E0; 

	eyieldp = sigyieldp/E0; 
	eyieldn = sigyieldn/E0;

	return 0;
}

UniaxialMaterial* SteelMPF::getCopy()
{
	UltParams ult{sigultp, sigultn, Rult0};
	SteelMPF* theCopy = new SteelMPF(this->getTag(), sigyieldp, sigyieldn,
		E0, bp, bn, R0, aa1, a2, a3, a4, a5, a6,
		hasSigUlt ? &ult : nullptr);

	// CONVERGED History variables
	theCopy-> incold = incold;

	theCopy-> Rptwoprevold = Rptwoprevold;
	theCopy-> Rntwoprevold = Rntwoprevold;

	theCopy-> outpold = outpold;
	theCopy-> outnold = outnold;

	theCopy-> erpold = erpold;
	theCopy-> sigrpold = sigrpold;

	theCopy-> ernold = ernold;
	theCopy-> sigrnold = sigrnold;

	theCopy-> erpmaxmaxold = erpmaxmaxold;
	theCopy-> ernmaxmaxold = ernmaxmaxold;

	theCopy-> e0pold = e0pold;
	theCopy-> sig0pold = sig0pold;

	theCopy-> e0nold = e0nold;
	theCopy-> sig0nold = sig0nold;

	theCopy-> erntwoprevold = erntwoprevold;
	theCopy-> sigrntwoprevold = sigrntwoprevold;
	theCopy-> e0ntwoprevold = e0ntwoprevold;
	theCopy-> sig0ntwoprevold = sig0ntwoprevold;

	theCopy-> erptwoprevold = erptwoprevold;
	theCopy-> sigrptwoprevold = sigrptwoprevold;
	theCopy-> e0ptwoprevold = e0ptwoprevold;
	theCopy-> sig0ptwoprevold = sig0ptwoprevold;

	theCopy-> Rpold = Rpold;
	theCopy-> Rnold = Rnold;

	theCopy-> nloopold = nloopold;

	// TRIAL History variables
	theCopy-> inc = inc;

	theCopy-> Rptwoprev = Rptwoprev;
	theCopy-> Rntwoprev = Rntwoprev;

	theCopy-> outp = outp;
	theCopy-> outn = outn;

	theCopy-> erp = erp;
	theCopy-> sigrp = sigrp;

	theCopy-> ern = ern;
	theCopy-> sigrn = sigrn;

	theCopy-> erpmaxmax = erpmaxmax;
	theCopy-> ernmaxmax = ernmaxmax;

	theCopy-> e0p = e0p;
	theCopy-> sig0p = sig0p;

	theCopy-> e0n = e0n;
	theCopy-> sig0n = sig0n;

	theCopy-> erntwoprev = erntwoprev;
	theCopy-> sigrntwoprev = sigrntwoprev;
	theCopy-> e0ntwoprev = e0ntwoprev;
	theCopy-> sig0ntwoprev = sig0ntwoprev;

	theCopy-> erptwoprev = erptwoprev;
	theCopy-> sigrptwoprev = sigrptwoprev;
	theCopy-> e0ptwoprev = e0ptwoprev;
	theCopy-> sig0ptwoprev = sig0ptwoprev;

	theCopy-> Rp = Rp;
	theCopy-> Rn = Rn;

	theCopy-> nloop = nloop;

	// CONVERGED State variables
	theCopy->defold = defold;
	theCopy->Fold = Fold;
	theCopy->stifold = stifold;

	// TRIAL State variables
	theCopy->def = def;
	theCopy->F = F;
	theCopy->stif = stif;

	return theCopy;
}

int SteelMPF::sendSelf (int commitTag, Channel& theChannel)
{
	int res = 0;
	static Vector data(46);

	data(0) = this->getTag();

	// Material properties (slots 1-12: unchanged from original layout)
	data(1) = sigyieldp;
	data(2) = sigyieldn;
	data(3) = E0;
	data(4) = bp;
	data(5) = bn;
	data(6) = R0;
	data(7) = aa1;
	data(8) = a2;
	data(9) = a3;
	data(10) = a4;
	data(11) = a5;
	data(12) = a6;

	// CONVERGED History variables (slots 13-38: unchanged from original layout)
	data(13) = incold;

	data(14) = Rptwoprevold;
	data(15) = Rntwoprevold;

	data(16) = outpold;
	data(17) = outnold;

	data(18) = erpold;
	data(19) = sigrpold;

	data(20) = ernold;
	data(21) = sigrnold;

	data(22) = erpmaxmaxold;
	data(23) = ernmaxmaxold;

	data(24) = e0pold;
	data(25) = sig0pold;

	data(26) = e0nold;
	data(27) = sig0nold;

	data(28) = erntwoprevold;
	data(29) = sigrntwoprevold;
	data(30) = e0ntwoprevold;
	data(31) = sig0ntwoprevold;

	data(32) = erptwoprevold;
	data(33) = sigrptwoprevold;
	data(34) = e0ptwoprevold;
	data(35) = sig0ptwoprevold;

	data(36) = Rpold;
	data(37) = Rnold;

	data(38) = nloopold;

	// CONVERGED State variables (slots 39-41: unchanged from original layout)
	data(39) = defold;
	data(40) = Fold;
	data(41) = stifold;

	// Ultimate stress cap parameters (slots 42-45: appended)
	data(42) = sigultp;
	data(43) = sigultn;
	data(44) = Rult0;
	data(45) = hasSigUlt ? 1.0 : 0.0;

	// Data is only sent after convergence, so no trial variables
	// need to be sent through data vector

	res = theChannel.sendVector(this->getDbTag(), commitTag, data);
	if (res < 0) 
		opserr << "SteelMPF::sendSelf() - failed to send data\n";

	return res;
}

int SteelMPF::recvSelf (int commitTag, Channel& theChannel, FEM_ObjectBroker& theBroker)
{
	int res = 0;

	static Vector data(46);
	res = theChannel.recvVector(this->getDbTag(), commitTag, data);

	if (res < 0) {
		opserr << "SteelMPF::recvSelf() - failed to receive data\n";
		this->setTag(0);      
	}
	else {
		this->setTag(int(data(0)));

		// Material properties (slots 1-12)
		sigyieldp = data(1);
		sigyieldn = data(2);
		E0 = data(3);
		bp = data(4);
		bn = data(5);
		R0 = data(6);
		aa1 = data(7);
		a2 = data(8);
		a3 = data(9);
		a4 = data(10);
		a5 = data(11);
		a6 = data(12);

		// CONVERGED History variables (slots 13-38)
		incold = int(data(13));

		Rptwoprevold = data(14);
		Rntwoprevold = data(15);

		outpold = int(data(16));
		outnold = int(data(17));

		erpold = data(18);
		sigrpold = data(19);

		ernold = data(20);
		sigrnold = data(21);

		erpmaxmaxold = data(22);
		ernmaxmaxold = data(23);

		e0pold = data(24);
		sig0pold = data(25);

		e0nold = data(26);
		sig0nold = data(27);

		erntwoprevold = data(28);
		sigrntwoprevold = data(29);
		e0ntwoprevold = data(30);
		sig0ntwoprevold = data(31);

		erptwoprevold = data(32);
		sigrptwoprevold = data(33);
		e0ptwoprevold = data(34);
		sig0ptwoprevold = data(35);

		Rpold = data(36);
		Rnold = data(37);

		nloopold = int(data(38));

		// CONVERGED State variables (slots 39-41)
		defold = data(39);
		Fold = data(40);
		stifold = data(41);

		// Ultimate stress cap parameters (slots 42-45)
		sigultp = data(42);
		sigultn = data(43);
		Rult0 = data(44);
		hasSigUlt = (data(45) > 0.5);

		// Copy converged history values into trial values since data is only
		// sent (received) after convergence
		inc = incold;

		Rptwoprev = Rptwoprevold;
		Rntwoprev = Rntwoprevold;

		outp = outpold;
		outn = outnold;

		erp = erpold;
		sigrp = sigrpold;

		ern = ernold;
		sigrn = sigrnold;

		erpmaxmax = erpmaxmaxold;
		ernmaxmax = ernmaxmaxold;

		e0p = e0pold;
		sig0p = sig0pold;

		e0n = e0nold;
		sig0n = sig0nold;

		erntwoprev = erntwoprevold;
		sigrntwoprev = sigrntwoprevold;
		e0ntwoprev = e0ntwoprevold;
		sig0ntwoprev = sig0ntwoprevold;

		erptwoprev = erptwoprevold;
		sigrptwoprev = sigrptwoprevold;
		e0ptwoprev = e0ptwoprevold;
		sig0ptwoprev = sig0ptwoprevold;

		Rp = Rpold;
		Rn = Rnold;

		nloop = nloopold;

		// Copy converged state values into trial values
		def = defold;
		F = Fold;
		stif = stifold;

	}

	return res; 
}

void SteelMPF::Print (OPS_Stream& s, int flag)
{
    if (flag == OPS_PRINT_PRINTMODEL_MATERIAL) {
        s << "SteelMPF tag: " << this->getTag() << endln;
        s << "fyp = " << sigyieldp << "\n";
        s << "fyn = " << sigyieldn << "\n";
        s << " E0 = " << E0 << "\n";
        s << " bp = " << bp << "\n";
        s << " bn = " << bn << "\n";
        s << "  R = " << R0 << "\n";
        s << "cR1 = " << aa1 << "\n";
        s << "cR2 = " << a2 << "\n";
        s << " a1 = " << a3 << "\n";
        s << " a2 = " << a4 << "\n";
        s << " a3 = " << a5 << "\n";
        s << " a4 = " << a6 << "\n";
        s << "fup = " << sigultp << "\n";
        s << "fun = " << sigultn << "\n";
        s << "Ru0 = " << Rult0 << "\n";
        s << "\n";
    }
    
    if (flag == OPS_PRINT_PRINTMODEL_JSON) {
        s << "\t\t\t{";
        s << "\"name\": \"" << this->getTag() << "\", ";
        s << "\"type\": \"SteelMPF\", ";
        s << "\"E\": " << E0 << ", ";
        s << "\"fyp\": " << sigyieldp << ", ";
        s << "\"fyn\": " << sigyieldn << ", ";
        s << "\"bp\": " << bp << ", ";
        s << "\"bn\": " << bn << ", ";
        s << "\"R0\": " << R0 << ", ";
        s << "\"cR1\": " << aa1 << ", ";
        s << "\"cR2\": " << a2 << ", ";
        s << "\"a1\": " << a3 << ", ";
        s << "\"a2\": " << a4 << ", ";
        s << "\"a3\": " << a5 << ", ";
        s << "\"a4\": " << a6 << ", ";
        s << "\"fup\": " << sigultp << ", ";
        s << "\"fun\": " << sigultn << ", ";
        s << "\"Ru0\": " << Rult0 << "}";
    }
}
