/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
** ****************************************************************** */

#ifndef ExplicitAlphaMultiSOE_TP_h
#define ExplicitAlphaMultiSOE_TP_h

#include <ExplicitAlphaMultiSOE.h>

/** Trailing-predictor variant of \c ExplicitAlphaMultiSOE (same sparse/workspace layout as the non-TP integrator).
 *  Like \c ExplicitAlpha_TP, two residual assemblies at \f$t_n\f$ and \f$t_{n+1}\f$ with residual weights are used to
 *  build a blended vector \c Put. The linear system keeps the primary operator \b A (same as \c ExplicitAlphaMultiSOE);
 *  \c formUnbalance uses \c Put in place of the usual equilibrium \f$\hat P\f$ and applies the same RHS transform as
 *  \c ExplicitAlphaMultiSOE::formUnbalance (\f$\alpha\,M^{-1}\,\texttt{Put}\f$ into \c soeA's \c B). */
class ExplicitAlphaMultiSOE_TP : public ExplicitAlphaMultiSOE
{
  public:
    ExplicitAlphaMultiSOE_TP();
    ExplicitAlphaMultiSOE_TP(double alphaF, double alphaM, double gamma, double beta,
                             bool incrementalAccel = false, bool useAlphaCloseCheck = false, bool order2 = false,
                             bool initAccel = false);
    ~ExplicitAlphaMultiSOE_TP() override;

    int formUnbalance(void) override;
    int formEleResidual(FE_Element *theEle) override;
    int formNodUnbalance(DOF_Group *theDof) override;

    int domainChanged(void) override;
    int newStep(double deltaT) override;
    int update(const Vector &aiPlusOne) override;
    int commit(void) override;

    const Vector &getVel(void) override;

    int sendSelf(int commitTag, Channel &theChannel) override;
    int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker) override;
    void Print(OPS_Stream &s, int flag = 0) override;
    int revertToStart(void) override;

  protected:
    ExplicitAlphaMultiSOE_TP(int classTag, double alphaF, double alphaM, double gamma, double beta,
                             bool incrementalAccel, bool useAlphaCloseCheck, bool order2 = false,
                             bool initAccel = false);

    Vector *Put;
    Vector *Vforce;  // -order2: velocity handed to nodal damping / getVel at the current force end
    double residM;
    double residD;
    double residR;
    double residP;
};

void *OPS_ExplicitAlphaMultiSOE_TP(void);
void *OPS_KRAlphaExplicitMultiSOE_TP(void);
void *OPS_MKRAlphaExplicitMultiSOE_TP(void);

#endif
