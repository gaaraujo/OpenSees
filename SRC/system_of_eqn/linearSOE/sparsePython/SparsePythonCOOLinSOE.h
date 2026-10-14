#ifndef SparsePythonCOOLinSOE_h
#define SparsePythonCOOLinSOE_h

#include "SparsePythonCommon.h"

#include <LinearSOE.h>
#include <Vector.h>

#include <vector>

class Channel;
class FEM_ObjectBroker;
class Graph;
class ID;
class Matrix;
class SparsePythonCOOLinSolver;

class SparsePythonCOOLinSOE : public LinearSOE
{
  public:
    using MatrixStatus = SparsePythonMatrixStatus;

    using MatrixType = SparsePythonMatrixType;

    explicit SparsePythonCOOLinSOE(SparsePythonCOOLinSolver &theSolver,
                                   MatrixType type = MatrixType::GENERAL);
    explicit SparsePythonCOOLinSOE(MatrixType type = MatrixType::GENERAL);
    ~SparsePythonCOOLinSOE() override;

    SparsePythonStorageScheme getStorageScheme() const { return SparsePythonStorageScheme::COO; }
    MatrixType getMatrixType() const { return theMatrixType; }

    int getNumEqn(void) const override;
    int setSize(Graph &theGraph) override;

    int addA(const Matrix &, const ID &, double fact = 1.0) override;
    int addB(const Vector &, const ID &, double fact = 1.0) override;
    int setB(const Vector &, double fact = 1.0) override;

    void zeroA(void) override;
    void zeroB(void) override;

    const Vector &getX(void) override;
    const Vector &getB(void) override;
    double normRHS(void) override;

    void setX(int loc, double value) override;
    void setX(const Vector &X) override;
    int solve(void) override;

    int sendSelf(int commitTag, Channel &theChannel) override;
    int recvSelf(int commitTag, Channel &theChannel, FEM_ObjectBroker &theBroker) override;

    int setPythonSolver(SparsePythonCOOLinSolver &newSolver);

    MatrixStatus getMatrixStatus(void) const { return matrixStatus; }
    void setMatrixStatus(MatrixStatus status) { matrixStatus = status; }

    /** Worker-side resize: allocate X/B only (no sparse pattern). */
    int resizeVectors(int size);

    const std::vector<int> &getRowIndices(void) const { return rowIndices; }
    const std::vector<int> &getColIndices(void) const { return colIndices; }
    std::vector<double> &getValues(void) { return values; }
    const std::vector<double> &getValues(void) const { return values; }
    std::vector<double> &getRHS(void) { return rhs; }
    const std::vector<double> &getRHS(void) const { return rhs; }
    std::vector<double> &getSolution(void) { return sol; }
    const std::vector<double> &getSolution(void) const { return sol; }

    std::size_t getNNZ(void) const { return values.size(); }

  private:
    void updateVectorViews();
    void buildCOOPattern(Graph &theGraph);

    std::vector<int> rowIndices;
    std::vector<int> colIndices;
    std::vector<double> values;
    std::vector<double> rhs;
    std::vector<double> sol;

    std::vector<int> rowOffsets;

    Vector xVector;
    Vector bVector;

    MatrixStatus matrixStatus;
    MatrixType theMatrixType;

    friend class SparsePythonCOOLinSolver;
};

#endif /* SparsePythonCOOLinSOE_h */
