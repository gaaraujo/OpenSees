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

#include <CudaExplicitAlpha.h>
#include <CudaBcsrLinSOE.h>
#include <DistributedCudaBcsrLinSOE.h>
#include <AnalysisModel.h>
#include <Channel.h>
#include <DOF_Group.h>
#include <DOF_GrpIter.h>
#include <FE_EleIter.h>
#include <FE_Element.h>
#include <FEM_ObjectBroker.h>
#include <Integrator.h>
#include <Matrix.h>
#include <OPS_Globals.h>
#include <LinearSOE.h>
#include <Vector.h>
#include <classTags.h>
#include <elementAPI.h>
#include <cmath>
#include <Domain.h>
#include <Element.h>
#include <Node.h>
#include <cstring>

#include "CudaCsrMatrix.h"
#include "CudaUtils.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/memory.h>
#include <vector>

using thrust::raw_pointer_cast;

using namespace CudaUtils;



// Kolay-Ricles midpoint-rule integrator on GPU (CudaBcsrLinSOE + cuDSS).
// Per step: GPU predictor -> equilibrium at t + alphaF*dt -> Linear solve -> update.
// Device scalar type T follows SOE precision (dDDI=double, dFFI=float); host stays double.

namespace {

// --- Templated device kernels ---

template<typename T>
__global__ void kernelAxpy(int n, T alpha, const T *x, T *y)
{
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        y[i] += alpha * x[i];
    }
}

// Midpoint predictor + alpha-interpolated response at t + alphaF*dt.
// alphaSol1 holds alpha^{-1} M Uddot_n; alphaSol2 holds alpha_3 Uddot_n when alphaM != alphaF.
template<typename T>
__global__ void kernelNewStep(int n, T dt, T gamma, T alphaF, T alphaM, const T *alphaSol2, T *U, T *Udot,
                              const T *Uddot, T *alphaSol1, T *Ualpha, T *Ualphadot, T *Ualphadotdot,
                              int incrementalAccel, int alphaClose)
{
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const T ut = U[i];
        const T utdot = Udot[i];
        const T utddot = Uddot[i];
        const T uhat = dt * alphaSol1[i];  // Udot_hat = dt * alpha_1 * Uddot_n
        alphaSol1[i] = uhat;
        const T un1 = ut + dt * utdot + (static_cast<T>(0.5) + gamma) * dt * uhat;
        const T udotn1 = utdot + uhat;
        U[i] = un1;
        Udot[i] = udotn1;
        // Alpha-level kinematics for equilibrium evaluation
        Ualpha[i] = (static_cast<T>(1.0) - alphaF) * ut + alphaF * un1;
        Ualphadot[i] = (static_cast<T>(1.0) - alphaF) * utdot + alphaF * udotn1;
        if (incrementalAccel) {
            Ualphadotdot[i] = utddot;
        } else if (alphaClose) {
            Ualphadotdot[i] = (static_cast<T>(1.0) - alphaM) * utddot;
        } else {
            Ualphadotdot[i] = utddot - alphaSol2[i];  // alpha_3 * Uddot_n
        }
    }
}

static int gridBlocks(int n, int blockSize = 256)
{
    return (n + blockSize - 1) / blockSize;
}

} // namespace

static constexpr int kMotionFields = 3;  // U, Udot, Uddot packed per node block

// Pimpl interface: double (host) / float|double (device) selected from SOE precision.
struct CudaExplicitAlpha::ImplBase {
    struct HostMotionPtrs {
        double *U;
        double *Udot;
        double *Uddot;
        double *Ut;       // step-start backup (maps to h_state_prev)
        double *Utdot;
        double *Utdotdot;
        double *Ualpha;   // response at t + alphaF*dt
        double *Ualphadot;
        double *Ualphadotdot;
    };

    virtual ~ImplBase() = default;

    virtual void allocate(int n, int alphaNumRhs) = 0;
    virtual void destroySolvers() = 0;
    virtual void shutdown() = 0;
    virtual void ensureAlphaBuffers(int alphaNumRhs) = 0;
    virtual int formOperators(CudaExplicitAlpha *integrator, CudaBcsrLinSOE *cudaSOE) = 0;
    virtual HostMotionPtrs ensureHostMotionBuffers(int n) = 0;
    virtual int newStepPredictor(CudaExplicitAlpha *integrator) = 0;
    virtual int formUnbalance(CudaBcsrLinSOE *cudaSOE) = 0;
    virtual int updateState(CudaBcsrLinSOE *cudaSOE, bool incrementalAccel) = 0;
    virtual int applyStart(CudaExplicitAlpha *integrator, CudaBcsrLinSOE *cudaSOE) = 0;
    virtual void zeroState() = 0;
    virtual int getSize() const = 0;
};

template<typename T>
struct ImplT : CudaExplicitAlpha::ImplBase {
    int size = 0;
    int alphaNumRhs = 2;  // 1 when alphaM ~= alphaF, or when incrementalAccel (alpha_3 unused)

    thrust::device_vector<T> d_state_cur;    // [U | Udot | Uddot]
    thrust::device_vector<T> d_state_alpha;  // [Ualpha | Ualphadot | Ualphadotdot]
    thrust::device_vector<T> d_w2;           // workspace for M^{-1} and solves
    thrust::device_vector<T> d_alphaRhs;     // RHS for alpha-operator solve(s)
    thrust::device_vector<T> d_alphaSol;     // alphaSol1 [| alphaSol2]
    pinned_host_vector<double> h_state_cur;    // [U | Udot | Uddot]
    pinned_host_vector<double> h_state_prev;   // [Ut | Utdot | Utdotdot] step-start backup
    pinned_host_vector<double> h_state_alpha;  // [Ualpha | Ualphadot | Ualphadotdot]

    cusparseHandle_t cusparseHandle = nullptr;
    cudaStream_t stream = nullptr;           // cached SOE stream from getCudaStream()
    CudaCsrMatrix *matM = nullptr;             // mass operator M
    CudaCsrMatrix *matAlpha = nullptr;         // alpha = M + gamma*dt*C + beta*dt^2*K
    CudaCsrMatrix *matA = nullptr;             // A = alphaM*M + alphaF*gamma*dt*C + alphaF*beta*dt^2*K
    int boundStructureRows = 0;
    int boundStructureNnz = 0;
    const int *boundRowPtr = nullptr;
    const int *boundColIdx = nullptr;

    int getSize() const override { return size; }

    T *devU() { return raw_pointer_cast(d_state_cur.data()); }
    T *devUdot() { return devU() + size; }
    T *devUddot() { return devU() + 2 * size; }
    const T *devU() const { return raw_pointer_cast(d_state_cur.data()); }
    const T *devUdot() const { return devU() + size; }
    const T *devUddot() const { return devU() + 2 * size; }
    T *devUalpha() { return raw_pointer_cast(d_state_alpha.data()); }
    T *devUalphadot() { return devUalpha() + size; }
    T *devUalphadotdot() { return devUalpha() + 2 * size; }

    void allocate(int n, int alphaNumRhsIn) override
    {
        size = n;
        alphaNumRhs = alphaNumRhsIn;
        const T zero = static_cast<T>(0.0);
        const std::size_t motionSize = static_cast<std::size_t>(kMotionFields * n);
        d_state_cur.assign(motionSize, zero);
        d_state_alpha.assign(motionSize, zero);
        d_w2.assign(n, zero);
        d_alphaRhs.assign(alphaNumRhs * n, zero);
        d_alphaSol.assign(alphaNumRhs * n, zero);
    }

    HostMotionPtrs ensureHostMotionBuffers(int n) override
    {
        HostMotionPtrs buf{};
        if (n <= 0) {
            return buf;
        }
        const std::size_t un = static_cast<std::size_t>(n);
        const std::size_t motionSize = static_cast<std::size_t>(kMotionFields) * un;
        h_state_cur.resize(motionSize);
        h_state_prev.resize(motionSize);
        h_state_alpha.resize(motionSize);
        double *cur = raw_pointer_cast(h_state_cur.data());
        double *prev = raw_pointer_cast(h_state_prev.data());
        double *alpha = raw_pointer_cast(h_state_alpha.data());
        buf.U = cur;
        buf.Udot = cur + n;
        buf.Uddot = cur + 2 * n;
        buf.Ut = prev;
        buf.Utdot = prev + n;
        buf.Utdotdot = prev + 2 * n;
        buf.Ualpha = alpha;
        buf.Ualphadot = alpha + n;
        buf.Ualphadotdot = alpha + 2 * n;
        return buf;
    }

    void ensureAlphaBuffers(int alphaNumRhsIn) override
    {
        if (alphaNumRhsIn == alphaNumRhs && static_cast<int>(d_alphaSol.size()) == alphaNumRhs * size) {
            return;
        }
        alphaNumRhs = alphaNumRhsIn;
        const T zero = static_cast<T>(0.0);
        d_alphaRhs.assign(alphaNumRhs * size, zero);
        d_alphaSol.assign(alphaNumRhs * size, zero);
    }

    void destroySolvers() override
    {
        if (matAlpha != nullptr) {
            matAlpha->detachSharedSpmvPattern();
        }
        if (matA != nullptr) {
            matA->detachSharedSpmvPattern();
        }
        delete matAlpha;
        delete matA;
        delete matM;
        matM = nullptr;
        matAlpha = nullptr;
        matA = nullptr;
        boundStructureRows = 0;
        boundStructureNnz = 0;
        boundRowPtr = nullptr;
        boundColIdx = nullptr;
    }

    void shutdown() override
    {
        if (stream != nullptr) {
            cudaStreamSynchronize(stream);
        }
        cudaDeviceSynchronize();
        destroySolvers();
        if (cusparseHandle != nullptr) {
            cusparseDestroy(cusparseHandle);
            cusparseHandle = nullptr;
        }
        stream = nullptr;
    }

    void makeMatrixConfigs(CudaPrecision prec, CudaCsrMatrix::SolverConfig &solver,
                           CudaCsrMatrix::SpmvConfig &spmv, CudaCsrMatrix::ExecutionContext &exec,
                           const CuSparseBackend *sharedPattern = nullptr) const
    {
        solver = CudaCsrMatrix::SolverConfig{};
        solver.precision = prec;
        solver.syncAfterSolve = false;
        exec.stream = stream;
        spmv.externalHandle = cusparseHandle;
        spmv.sharedPattern = sharedPattern;
    }

    int bindStream(CudaBcsrLinSOE *cudaSOE)
    {
        if (cudaSOE == nullptr) {
            return -1;
        }
        stream = static_cast<cudaStream_t>(cudaSOE->getCudaStream());
        return stream == nullptr ? -1 : 0;
    }

    void initCusparse()
    {
        if (cusparseHandle == nullptr) {
            cuSparseCheckError(cusparseCreate(&cusparseHandle), "cusparseCreate");
            cuSparseCheckError(cusparseSetStream(cusparseHandle, stream), "cusparseSetStream");
        }
    }

    void ensureMatrices(CudaPrecision prec)
    {
        initCusparse();
        if (matM == nullptr) {
            CudaCsrMatrix::SolverConfig solver;
            CudaCsrMatrix::SpmvConfig spmv;
            CudaCsrMatrix::ExecutionContext exec;
            makeMatrixConfigs(prec, solver, spmv, exec);
            matM = new CudaCsrMatrix(solver, spmv, exec);

            makeMatrixConfigs(prec, solver, spmv, exec, matM->getSpmvBackend());
            matAlpha = new CudaCsrMatrix(solver, spmv, exec);

            makeMatrixConfigs(prec, solver, spmv, exec, matM->getSpmvBackend());
            matA = new CudaCsrMatrix(solver, spmv, exec);
        }
    }

    int bindSharedStructure(CudaBcsrLinSOE *soe)
    {
        // Reuse one CSR pattern for matM, matAlpha, and matA (values differ).
        const int nnz = soe->getNumNonZeroValues();
        const int *rowPtr = soe->getDeviceRowPtrs();
        const int *colIdx = soe->getDeviceColIndices();
        if (rowPtr == nullptr || colIdx == nullptr) {
            opserr << "ERROR CudaExplicitAlpha::bindSharedStructure - null device CSR indices "
                   << "(size=" << size << " nnz=" << nnz << " rowPtr=" << rowPtr
                   << " colIdx=" << colIdx << ")\n";
            return -1;
        }

        const bool samePattern = boundStructureRows == size && boundStructureNnz == nnz && boundRowPtr == rowPtr &&
                                 boundColIdx == colIdx && matM != nullptr && matM->isStructureBound();

        if (!samePattern) {
            if (matM != nullptr && matM->isStructureBound()) {
                matM->reset();
                matAlpha->reset();
                matA->reset();
            }
            if (matM->bindStructure(size, nnz, rowPtr, colIdx) != 0) {
                opserr << "ERROR CudaExplicitAlpha::bindSharedStructure - matM->bindStructure failed\n";
                return -1;
            }
            if (matAlpha->bindStructure(size, nnz, rowPtr, colIdx) != 0) {
                opserr << "ERROR CudaExplicitAlpha::bindSharedStructure - matAlpha->bindStructure failed\n";
                return -1;
            }
            if (matA->bindStructure(size, nnz, rowPtr, colIdx) != 0) {
                opserr << "ERROR CudaExplicitAlpha::bindSharedStructure - matA->bindStructure failed\n";
                return -1;
            }
            boundStructureRows = size;
            boundStructureNnz = nnz;
            boundRowPtr = rowPtr;
            boundColIdx = colIdx;
        }
        return 0;
    }

    int applyM(const T *x, T *y) { return matM->spmv(x, y); }

    int formOperators(CudaExplicitAlpha *integrator, CudaBcsrLinSOE *cudaSOE) override
    {
        // Build/refactor GPU operators used by newStepPredictor() and formUnbalance().
        // Critical MPI ordering: finish ALL three collective formTangentIntoSOE /
        // ensureMergedA exchanges BEFORE any long GPU factorize. Factorizing between
        // merges leaves worker triplet messages buffered in the MPI layer; with
        // np>=4 (several large Auto/penalty triplet payloads) that has produced
        // silently wrong operators and an exploding predictor. MultiSOE avoids this
        // by merging once per workspace SOE inside solve/formAp, not around GPU work.
        const double bdt2 = integrator->beta * integrator->deltaT * integrator->deltaT;
        const double gdt = integrator->gamma * integrator->deltaT;
        const bool areClose = integrator->areAlphaMFClose();
        // alpha_3 column is only consumed by the non-incremental predictor kernel.
        // Match ExplicitAlphaMultiSOE::newStep, which skips applyAlpha3 under -incrementalAccel.
        const int numRhs = (areClose || integrator->incrementalAccel) ? 1 : 2;
        int gpuRc = 0;

        ensureAlphaBuffers(numRhs);

        void *rhsM = thrust::raw_pointer_cast(d_alphaRhs.data());
        void *solM = thrust::raw_pointer_cast(d_w2.data());
        void *rhsAlpha = rhsM;
        void *solAlpha1 = thrust::raw_pointer_cast(d_alphaSol.data());

        // --- Mass operator M (collective assemble+merge, then fast D2D snapshot) ---
        if (integrator->formTangentIntoSOE(INITIAL_TANGENT, 0.0, 0.0, 1.0) != 0)
            return -1;

        if (gpuRc == 0 && bindStream(cudaSOE) != 0) {
            opserr << "ERROR CudaExplicitAlpha::formOperators - bindStream failed\n";
            gpuRc = -1;
        }
        if (gpuRc == 0) {
            cudaSOE->syncIndicesToDevice();
            cudaSOE->ensureDeviceVectorSizes();
            ensureMatrices(cudaSOE->getPrecision());
            if (bindSharedStructure(cudaSOE) != 0)
                gpuRc = -1;
        }
        if (gpuRc == 0) {
            cudaSOE->syncAValuesToDevice();
            if (matM->copyValues(cudaSOE->getDeviceAValues()) != 0) {
                opserr << "ERROR CudaExplicitAlpha::formOperators - matM->copyValues failed\n";
                gpuRc = -2;
            }
        }

        // --- A_alpha (collective; snapshot only — factorize later) ---
        if (integrator->formTangentIntoSOE(INITIAL_TANGENT, bdt2, gdt, 1.0) != 0)
            return -3;
        if (gpuRc == 0) {
            cudaSOE->syncAValuesToDevice();
            if (matAlpha->copyValues(cudaSOE->getDeviceAValues()) != 0) {
                opserr << "ERROR CudaExplicitAlpha::formOperators - matAlpha->copyValues failed\n";
                gpuRc = -4;
            }
        }

        // --- A / Ã (collective; snapshot only) ---
        if (areClose) {
            if (integrator->formTangentIntoSOE(INITIAL_TANGENT, 0.0, 0.0, integrator->alphaM) != 0)
                return -5;
        } else if (integrator->formTangentIntoSOE(INITIAL_TANGENT, integrator->alphaF * bdt2,
                                                  integrator->alphaF * gdt, integrator->alphaM) != 0) {
            return -5;
        }
        if (gpuRc == 0) {
            cudaSOE->syncAValuesToDevice();
            if (matA->copyValues(cudaSOE->getDeviceAValues()) != 0) {
                opserr << "ERROR CudaExplicitAlpha::formOperators - matA->copyValues failed\n";
                gpuRc = -5;
            }
        }

        // --- GPU factorize AFTER all MPI merges are complete ---
        if (gpuRc == 0) {
            if (!matAlpha->isFactored()) {
                if (matAlpha->factorize(rhsAlpha, solAlpha1, numRhs, nullptr) != 0) {
                    opserr << "ERROR CudaExplicitAlpha::formOperators - matAlpha->factorize failed\n";
                    gpuRc = -4;
                }
            } else if (matAlpha->refactorize(rhsAlpha, solAlpha1, numRhs) != 0) {
                opserr << "ERROR CudaExplicitAlpha::formOperators - matAlpha->refactorize failed\n";
                gpuRc = -4;
            }
        }
        // Factorize M independently. Sharing alpha's symbolic factorization has
        // been observed to produce huge M^{-1} amplification under Auto penalties
        // at np>=4 even when ||M|| matches the working np=3 assemble.
        if (gpuRc == 0) {
            if (!matM->isFactored()) {
                if (matM->factorize(rhsM, solM, 1, nullptr) != 0) {
                    opserr << "ERROR CudaExplicitAlpha::formOperators - matM->factorize failed\n";
                    gpuRc = -2;
                }
            } else if (matM->refactorize(rhsM, solM) != 0) {
                opserr << "ERROR CudaExplicitAlpha::formOperators - matM->refactorize failed\n";
                gpuRc = -2;
            }
        }
        return gpuRc;
    }

    // GPU predictor: alpha_1 solve, then midpoint + alpha-interpolated kinematics.
    int newStepPredictor(CudaExplicitAlpha *integrator) override
    {
        // Steady steps already hold committed (U,Udot,Uddot) on the device from updateState.
        // Re-upload only after host-only writes (domainChanged / revert / DistCuDSS gather).
        if (size > 0 && integrator->deviceMotionStale) {
            d_state_cur = h_state_cur;
            integrator->deviceMotionStale = false;
        }
        T *dU = devU();
        T *dUdot = devUdot();
        T *dUddot = devUddot();
        T *dAlphaSol1 = thrust::raw_pointer_cast(d_alphaSol.data());
        T *dAlphaSol2 = alphaNumRhs > 1 ? dAlphaSol1 + size : nullptr;
        T *dUalpha = devUalpha();
        T *dUalphadot = devUalphadot();
        T *dUalphadotdot = devUalphadotdot();
        T *dAlphaRhs = thrust::raw_pointer_cast(d_alphaRhs.data());

        const T dt = static_cast<T>(integrator->deltaT);
        const T gammaT = static_cast<T>(integrator->gamma);
        const T alphaFT = static_cast<T>(integrator->alphaF);
        const T alphaMT = static_cast<T>(integrator->alphaM);
        const bool alphaClose = integrator->areAlphaMFClose();
        const int incAccel = integrator->incrementalAccel ? 1 : 0;
        // Need alpha_3 = I - alpha^{-1} A only for non-incremental trial accel when alphaM != alphaF.
        const bool needAlpha3 = (incAccel == 0) && !alphaClose;

        applyM(dUddot, dAlphaRhs);
        if (needAlpha3) {
            matA->spmv(dUddot, dAlphaRhs + size);
            if (matAlpha->solve(dAlphaRhs, dAlphaSol1, alphaNumRhs) != 0) {
                return -5;
            }
            kernelNewStep<<<gridBlocks(size), 256, 0, stream>>>(size, dt, gammaT, alphaFT, alphaMT, dAlphaSol2, dU,
                                                                dUdot, dUddot, dAlphaSol1, dUalpha, dUalphadot,
                                                                dUalphadotdot, incAccel, 0);
        } else {
            if (matAlpha->solve(dAlphaRhs, dAlphaSol1, 1) != 0) {
                return -5;
            }
            kernelNewStep<<<gridBlocks(size), 256, 0, stream>>>(
                size, dt, gammaT, alphaFT, alphaMT, static_cast<const T *>(nullptr), dU, dUdot, dUddot, dAlphaSol1,
                dUalpha, dUalphadot, dUalphadotdot, incAccel, alphaClose ? 1 : 0);
        }
        // All predictor ops share integrator stream; sync once before Thrust D2H host mirror update.
        cudaCheckError(cudaStreamSynchronize(stream), "integrator stream sync before host read");
        // Host U/Udot are updated to n+1 here — required before OpenSeesMP broadcastFromRoot
        // so workers (who may own recorded nodes) setDisp the predicted displacement.
        h_state_cur = d_state_cur;
        h_state_alpha = d_state_alpha;
        return 0;
    }

    // -initAccel start: host B holds R0 = p - C v0 - f(u0, v0) (zero trial accel).
    // a0 = M^{-1} R0. Host mirror refreshed.
    int applyStart(CudaExplicitAlpha *integrator, CudaBcsrLinSOE *cudaSOE) override
    {
        if (matM == nullptr || !matM->isFactored()) {
            return -1;
        }
        if (size > 0 && integrator->deviceMotionStale) {
            d_state_cur = h_state_cur;
            integrator->deviceMotionStale = false;
        }
        cudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
        cudaSOE->syncBToDevice();
        void *b = cudaSOE->getDeviceB();
        if (matM->solve(b, devUddot()) != 0) {
            return -2;
        }
        cudaCheckError(cudaStreamSynchronize(stream), "integrator stream sync after -initAccel start");
        h_state_cur = d_state_cur;
        return 0;
    }

    // Called after host equilibrium assembly: B <- alpha * M^{-1} * Phat.
    int formUnbalance(CudaBcsrLinSOE *cudaSOE) override
    {
        cudaSOE->syncBToDevice();
        void *b = cudaSOE->getDeviceB();
        T *w2 = thrust::raw_pointer_cast(d_w2.data());
        if (matM == nullptr || !matM->isFactored()) {
            return -3;
        }
        if (matM->solve(b, w2) != 0) {
            return -3;
        }
        matAlpha->spmv(w2, b);
        cudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Device);
        return 0;
    }

    // Copy solved acceleration from SOE X into device state; refresh host Uddot only.
    // U/Udot are already on the host from newStepPredictor (and were not changed here).
    int updateState(CudaBcsrLinSOE *cudaSOE, bool incrementalAccel) override
    {
        T *dUddot = devUddot();
        const T *dX = static_cast<const T *>(cudaSOE->getDeviceX());
        if (dX == nullptr) {
            return -4;
        }
        if (incrementalAccel) {
            const T one = static_cast<T>(1.0);
            kernelAxpy<<<gridBlocks(size), 256, 0, stream>>>(size, one, dX, dUddot);
        } else {
            cudaMemcpyAsync(dUddot, dX, static_cast<std::size_t>(size) * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        }
        cudaCheckError(cudaStreamSynchronize(stream), "integrator stream sync before host read");

        double *hUddot = raw_pointer_cast(h_state_cur.data()) + 2 * size;
        if (incrementalAccel) {
            // Pull only accel (n) so host matches GPU float/double accumulation.
            thrust::copy(d_state_cur.begin() + 2 * size, d_state_cur.begin() + 3 * size,
                         h_state_cur.begin() + 2 * size);
        } else {
            // X already on host after DistCuDSS::solve (or sync here for serial CuDSS).
            cudaSOE->syncXToHost();
            const Vector &x = cudaSOE->getHostXVector();
            for (int i = 0; i < size; ++i)
                hUddot[i] = x(i);
        }
        return 0;
    }

    void zeroState() override
    {
        if (size <= 0) {
            return;
        }
        const T zero = static_cast<T>(0.0);
        d_state_cur.assign(static_cast<std::size_t>(kMotionFields * size), zero);
    }
};

template struct ImplT<double>;
template struct ImplT<float>;

void CudaExplicitAlpha::destroyDeviceImpl()
{
    if (m_impl != nullptr) {
        m_impl->shutdown();
        delete m_impl;
        m_impl = nullptr;
    }
}

void CudaExplicitAlpha::ensureDeviceImpl(CudaBcsrLinSOE *cudaSOE)
{
    // Pick ImplT<float> or ImplT<double> to match uniform SOE precision.
    const CudaPrecision prec = cudaSOE->getPrecision();
    const bool useFloat = (prec == CudaPrecision::dFFI);
    if (m_impl != nullptr && m_deviceUsesFloat == useFloat) {
        return;
    }
    destroyDeviceImpl();
    if (useFloat) {
        m_impl = new ImplT<float>();
    } else {
        m_impl = new ImplT<double>();
    }
    m_deviceUsesFloat = useFloat;
}

int CudaExplicitAlpha::formOperators(CudaBcsrLinSOE *cudaSOE)
{
    const double bdt2 = beta * deltaT * deltaT;
    const double gdt = gamma * deltaT;
    const bool areClose = areAlphaMFClose();
    auto *distSOE = dynamic_cast<DistributedCudaBcsrLinSOE *>(this->getLinearSOE());

    // OpenSeesMP workers: same three collective assemblies as rank 0, then wait for
    // root's GPU factorize via a status broadcast (avoids hanging in newStep broadcast).
    if (!cudaSOE->isCudaDeviceEnabled()) {
        if (formTangentIntoSOE(INITIAL_TANGENT, 0.0, 0.0, 1.0) != 0)
            return -1;
        if (formTangentIntoSOE(INITIAL_TANGENT, bdt2, gdt, 1.0) != 0)
            return -3;
        if (areClose) {
            if (formTangentIntoSOE(INITIAL_TANGENT, 0.0, 0.0, alphaM) != 0)
                return -5;
        } else if (formTangentIntoSOE(INITIAL_TANGENT, alphaF * bdt2, alphaF * gdt, alphaM) != 0) {
            return -5;
        }
        if (distSOE != nullptr) {
            Vector status(1);
            if (distSOE->broadcastFromRoot(status) < 0)
                return -6;
            if (status(0) < 0.0) {
                opserr << "ERROR CudaExplicitAlpha::formOperators - rank 0 GPU operator build failed\n";
                return -6;
            }
        }
        return 0;
    }

    if (m_impl == nullptr)
        return -1;
    const int rc = m_impl->formOperators(this, cudaSOE);
    if (distSOE != nullptr) {
        Vector status(1);
        status(0) = (rc == 0) ? 1.0 : -1.0;
        if (distSOE->broadcastFromRoot(status) < 0)
            return -6;
    }
    return rc;
}

namespace {

bool parseCudaExplicitAlphaOptions(CudaExplicitAlpha::Options &opts)
{
    opts = CudaExplicitAlpha::Options{};
    while (OPS_GetNumRemainingInputArgs() > 0) {
        const char *tok = OPS_GetString();
        if (tok == nullptr) {
            break;
        }
        if (strcmp(tok, "-updateElemDisp") == 0) {
            opts.updElemDisp = true;
        } else if (strcmp(tok, "-incrementalAccel") == 0) {
            opts.incrementalAccel = true;
        } else if (strcmp(tok, "-alphaCloseCheck") == 0) {
            opts.useAlphaCloseCheck = true;
        } else if (strcmp(tok, "-initAccel") == 0) {
            opts.initAccel = true;
        } else {
            opserr << "WARNING CudaExplicitAlpha family - unknown flag " << tok
                   << "; want <-updateElemDisp> <-incrementalAccel> <-alphaCloseCheck> <-initAccel>\n";
            return false;
        }
    }
    return true;
}

} // namespace

void *OPS_CudaExplicitAlpha(void)
{
    if (OPS_GetNumRemainingInputArgs() < 4) {
        opserr << "WARNING integrator CudaExplicitAlpha alphaF alphaM gamma beta "
                  "<-updateElemDisp> <-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }
    double alphaF = 0.0;
    double alphaM = 0.0;
    double gamma = 0.0;
    double beta = 0.0;
    int numData = 1;
    if (OPS_GetDoubleInput(&numData, &alphaF) != 0 || OPS_GetDoubleInput(&numData, &alphaM) != 0 ||
        OPS_GetDoubleInput(&numData, &gamma) != 0 || OPS_GetDoubleInput(&numData, &beta) != 0) {
        opserr << "WARNING CudaExplicitAlpha - invalid alphaF/alphaM/gamma/beta\n";
        return nullptr;
    }
    CudaExplicitAlpha::Options opts;
    if (!parseCudaExplicitAlphaOptions(opts)) {
        return nullptr;
    }
    return new CudaExplicitAlpha(alphaF, alphaM, gamma, beta, opts);
}

void *OPS_CudaKRAlpha(void)
{
    if (OPS_GetNumRemainingInputArgs() < 1) {
        opserr << "WARNING integrator CudaKRAlpha rhoInf "
                  "<-updateElemDisp> <-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }
    double rhoInf = 0.0;
    int numData = 1;
    if (OPS_GetDoubleInput(&numData, &rhoInf) != 0) {
        opserr << "WARNING CudaKRAlpha - invalid rhoInf\n";
        return nullptr;
    }
    const double den = 1.0 + rhoInf;
    const double alphaF = 1.0 / den;
    const double alphaM = (2.0 - rhoInf) / den;
    const double gamma = 0.5 * (3.0 - rhoInf) / den;
    const double beta = 1.0 / (den * den);
    CudaExplicitAlpha::Options opts;
    if (!parseCudaExplicitAlphaOptions(opts)) {
        return nullptr;
    }
    return new CudaExplicitAlpha(alphaF, alphaM, gamma, beta, opts);
}

void *OPS_CudaMKRAlpha(void)
{
    if (OPS_GetNumRemainingInputArgs() < 1) {
        opserr << "WARNING integrator CudaMKRAlpha rhoInfEquivalent "
                  "<-updateElemDisp> <-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }
    double rhoInfEquivalent = 0.0;
    int numData = 1;
    if (OPS_GetDoubleInput(&numData, &rhoInfEquivalent) != 0) {
        opserr << "WARNING CudaMKRAlpha - invalid rhoInfEquivalent\n";
        return nullptr;
    }
    const double r = (rhoInfEquivalent >= 0.0) ? std::sqrt(rhoInfEquivalent) : rhoInfEquivalent;
    const double alphaF = 1.0 / (r + 1.0);
    const double r2 = r * r;
    const double r3 = r2 * r;
    const double num = (2.0 * r3 + r2 - 1.0);
    const double den = (r3 + r2 + r + 1.0);
    const double alphaM = 1.0 - num / den;
    const double gamma = alphaM - alphaF + 0.5;
    const double beta = 0.5 * (gamma + 0.5);
    CudaExplicitAlpha::Options opts;
    if (!parseCudaExplicitAlphaOptions(opts)) {
        return nullptr;
    }
    return new CudaExplicitAlpha(alphaF, alphaM, gamma, beta, opts);
}

CudaExplicitAlpha::CudaExplicitAlpha()
    : CudaExplicitAlpha(INTEGRATOR_TAGS_CudaExplicitAlpha, 0.5, 0.5, 0.5, 0.25, Options{})
{
}

CudaExplicitAlpha::CudaExplicitAlpha(double _alphaF, double _alphaM, double _gamma, double _beta)
    : CudaExplicitAlpha(_alphaF, _alphaM, _gamma, _beta, Options{})
{
}

CudaExplicitAlpha::CudaExplicitAlpha(double _alphaF, double _alphaM, double _gamma, double _beta, const Options &opts)
    : CudaExplicitAlpha(INTEGRATOR_TAGS_CudaExplicitAlpha, _alphaF, _alphaM, _gamma, _beta, opts)
{
}

CudaExplicitAlpha::CudaExplicitAlpha(int classTag, double _alphaF, double _alphaM, double _gamma, double _beta,
                                     const Options &opts)
    : TransientIntegrator(classTag),
      alphaM(_alphaM),
      alphaF(_alphaF),
      beta(_beta),
      gamma(_gamma),
      updElemDisp(opts.updElemDisp),
      incrementalAccel(opts.incrementalAccel),
      useAlphaCloseCheck(opts.useAlphaCloseCheck),
      initAccel(opts.initAccel),
      startPending(true),
      tStepStart(-1.0),
      updateCount(0),
      c1(0.0),
      c2(0.0),
      c3(0.0),
      deltaT(0.0),
      Ut(new Vector(0)),
      Utdot(new Vector(0)),
      Utdotdot(new Vector(0)),
      U(new Vector(0)),
      Udot(new Vector(0)),
      Udotdot(new Vector(0)),
      Ualpha(new Vector(0)),
      Ualphadot(new Vector(0)),
      Ualphadotdot(new Vector(0)),
      operatorsBuilt(false),
      motionNeedsGather(true),
      deviceMotionStale(true),
      m_impl(nullptr)
{
}

CudaExplicitAlpha::~CudaExplicitAlpha()
{
    CudaBcsrLinSOE *cudaSOE = nullptr;
    if (validateCudaSOE(cudaSOE) == 0) {
        cudaSOE->setXSyncMode(true);
    }
    destroyDeviceImpl();
    delete Ut;
    delete Utdot;
    delete Utdotdot;
    delete U;
    delete Udot;
    delete Udotdot;
    delete Ualpha;
    delete Ualphadot;
    delete Ualphadotdot;
}

bool CudaExplicitAlpha::areAlphaMFClose() const
{
    return useAlphaCloseCheck && std::fabs(alphaM - alphaF) <= toleranceAlphaMF;
}

int CudaExplicitAlpha::validateCudaSOE(CudaBcsrLinSOE *&cudaSOE) const
{
    LinearSOE *soe = this->getLinearSOE();
    if (soe == nullptr) {
        opserr << "ERROR CudaExplicitAlpha - no LinearSOE linked; use `system CuDSS` or `system DistributedCuDSS`.\n";
        return -1;
    }
    cudaSOE = dynamic_cast<CudaBcsrLinSOE *>(soe);
    if (cudaSOE == nullptr) {
        if (auto *dist = dynamic_cast<DistributedCudaBcsrLinSOE *>(soe))
            cudaSOE = dist->getCudaBcsrLinSOE();
    }
    if (cudaSOE == nullptr || cudaSOE->getBlockSize() != 1 || cudaSOE->getCudaBcsrLinSolver() == nullptr) {
        opserr << "ERROR CudaExplicitAlpha (CudaKRAlpha/CudaMKRAlpha) requires `system CuDSS` or `system DistributedCuDSS`.\n";
        return -2;
    }
    if (!isUniformPrecision(cudaSOE->getPrecision())) {
        opserr << "ERROR CudaExplicitAlpha - unsupported CuDSS precision "
               << cudaPrecisionToString(cudaSOE->getPrecision())
               << "; only uniform dDDI and dFFI are supported.\n";
        return -3;
    }
    return 0;
}

int CudaExplicitAlpha::formTangentIntoSOE(int statFlag, double c1v, double c2v, double c3v)
{
    if (this->getAnalysisModel() == nullptr) {
        return -1;
    }
    LinearSOE *soe = this->getLinearSOE();
    if (soe == nullptr) {
        return -1;
    }
    // Zero/assemble through the linked SOE (Distributed wrapper when present) so
    // worker triplets participate; then flush them into rank-0 CSR if needed.
    soe->zeroA();
    statusFlag = statFlag;
    c1 = c1v;
    c2 = c2v;
    c3 = c3v;
    if (this->TransientIntegrator::formTangent(statFlag) != 0) {
        return -1;
    }
    if (auto *dist = dynamic_cast<DistributedCudaBcsrLinSOE *>(soe)) {
        if (dist->ensureMergedA() != 0) {
            return -1;
        }
        // Merged coefficients live on host; force Host authority so the next
        // syncAValuesToDevice cannot skip a stale Both/device snapshot.
        if (CudaBcsrLinSOE *cuda = dist->getCudaBcsrLinSOE()) {
            cuda->setAValuesPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
        }
    }
    return 0;
}

int CudaExplicitAlpha::formTangent(int statFlag)
{
    (void)statFlag;
    return 0;  // operators assembled in formOperators(); Linear::formTangent is a no-op
}

int CudaExplicitAlpha::formEleTangent(FE_Element *theEle)
{
    if (operatorsBuilt) {
        return 0;
    }
    theEle->zeroTangent();
    if (statusFlag == CURRENT_TANGENT) {
        theEle->addKtToTang(c1);
    } else if (statusFlag == INITIAL_TANGENT) {
        theEle->addKiToTang(c1);
    }
    theEle->addCtoTang(c2);
    theEle->addMtoTang(c3);
    return 0;
}

int CudaExplicitAlpha::formNodTangent(DOF_Group *theDof)
{
    if (operatorsBuilt) {
        return 0;
    }
    theDof->zeroTangent();
    theDof->addCtoTang(c2);
    theDof->addMtoTang(c3);
    return 0;
}

int CudaExplicitAlpha::domainChanged()
{
    CudaBcsrLinSOE *cudaSOE = nullptr;
    if (validateCudaSOE(cudaSOE) != 0) {
        operatorsBuilt = false;
        return -1;
    }
    const int size = cudaSOE->getNumEqn();
    if (size <= 0) {
        operatorsBuilt = false;
        return 0;
    }

    const bool deviceOn = cudaSOE->isCudaDeviceEnabled();
    if (deviceOn) {
        ensureDeviceImpl(cudaSOE);
        cudaSOE->setXSyncMode(false);  // keep X on device between solve and updateState
        m_impl->destroySolvers();
        m_impl->allocate(size, (areAlphaMFClose() || incrementalAccel) ? 1 : 2);
        const ImplBase::HostMotionPtrs buf = m_impl->ensureHostMotionBuffers(size);
        Ut->setData(buf.Ut, size);
        Utdot->setData(buf.Utdot, size);
        Utdotdot->setData(buf.Utdotdot, size);
        U->setData(buf.U, size);
        Udot->setData(buf.Udot, size);
        Udotdot->setData(buf.Uddot, size);
        Ualpha->setData(buf.Ualpha, size);
        Ualphadot->setData(buf.Ualphadot, size);
        Ualphadotdot->setData(buf.Ualphadotdot, size);
    } else {
        // OpenSeesMP workers: host-only kinematics; GPU lives on rank 0.
        destroyDeviceImpl();
        Ut->resize(size); Utdot->resize(size); Utdotdot->resize(size);
        U->resize(size); Udot->resize(size); Udotdot->resize(size);
        Ualpha->resize(size); Ualphadot->resize(size); Ualphadotdot->resize(size);
        Ut->Zero(); Utdot->Zero(); Utdotdot->Zero();
        U->Zero(); Udot->Zero(); Udotdot->Zero();
        Ualpha->Zero(); Ualphadot->Zero(); Ualphadotdot->Zero();
    }
    operatorsBuilt = false;

    // Seed motion vectors from committed domain state.
    AnalysisModel *theModel = this->getAnalysisModel();
    DOF_GrpIter &theDOFs = theModel->getDOFs();
    DOF_Group *dofPtr;
    while ((dofPtr = theDOFs()) != nullptr) {
        const ID &id = dofPtr->getID();
        const int idSize = id.Size();
        // Copy each vector before the next getter. TransformationDOF_Group
        // writes disp, vel, and accel into the same modUnbalance buffer.
        const Vector &disp = dofPtr->getCommittedDisp();
        for (int i = 0; i < idSize; ++i) {
            const int loc = id(i);
            if (loc >= 0)
                (*U)(loc) = disp(i);
        }
        const Vector &vel = dofPtr->getCommittedVel();
        for (int i = 0; i < idSize; ++i) {
            const int loc = id(i);
            if (loc >= 0)
                (*Udot)(loc) = vel(i);
        }
        const Vector &accel = dofPtr->getCommittedAccel();
        for (int i = 0; i < idSize; ++i) {
            const int loc = id(i);
            if (loc >= 0)
                (*Udotdot)(loc) = accel(i);
        }
    }
    *Ut = *U;
    *Utdot = *Udot;
    *Utdotdot = *Udotdot;
    motionNeedsGather = true;
    deviceMotionStale = true;
    return 0;
}

// -initAccel starting correction (collective in OpenSeesMP): assemble R0 = p - C v0 - f(u0, v0)
// with zero trial accel, merge to root, then a0 = M^{-1} R0 on the root GPU.
int CudaExplicitAlpha::applyStartCorrection(CudaBcsrLinSOE *cudaSOE)
{
    AnalysisModel *theModel = this->getAnalysisModel();
    auto *distSOE = dynamic_cast<DistributedCudaBcsrLinSOE *>(this->getLinearSOE());
    Ualphadotdot->Zero();
    // Elements that update from the trial-displacement increment (forceBeamColumn) skip their update
    // when the increment is zero, so after setNodeDisp -commit they still hold the previous state and
    // R0 would miss the imposed displacement. A tiny non-committing trial excursion makes them iterate
    // to the committed configuration before R0 is formed.
    {
        Vector Up(*U);
        for (int i = 0; i < Up.Size(); i++)
            Up(i) += 1.0e-9 * (1.0 + fabs(Up(i)));
        theModel->setResponse(Up, *Udot, *Ualphadotdot);
        if (theModel->updateDomain(theModel->getCurrentDomainTime(), deltaT) < 0)
            return -2;
    }
    theModel->setResponse(*U, *Udot, *Ualphadotdot);
    if (theModel->updateDomain(theModel->getCurrentDomainTime(), deltaT) < 0)
        return -1;
    if (TransientIntegrator::formUnbalance() < 0)
        return -2;

    int rc = 0;
    if (distSOE != nullptr) {
        rc = distSOE->mergeBToRoot();
        if (rc == 0)
            distSOE->zeroB();
        Vector status(1);
        if (distSOE->getProcessID() != 0) {
            if (distSOE->broadcastFromRoot(status) < 0)
                return -3;
            return (status(0) < 0.0) ? -4 : 0;
        }
        if (rc == 0)
            rc = (m_impl != nullptr) ? m_impl->applyStart(this, cudaSOE) : -5;
        status(0) = (rc == 0) ? 1.0 : -1.0;
        if (distSOE->broadcastFromRoot(status) < 0)
            return -3;
        return rc;
    }
    if (m_impl == nullptr)
        return -5;
    return m_impl->applyStart(this, cudaSOE);
}

// Per-step flow: GPU predictor -> setResponse at alpha level -> updateDomain at t + alphaF*dt.
// Linear::solveCurrentStep then formUnbalance (equilibrium + GPU RHS) and update (full kinematics).
int CudaExplicitAlpha::newStep(double _deltaT)
{
    updateCount = 0;
    if (alphaF < 0.5 || alphaF > 1.0 || beta <= 0.0 || gamma <= 0.0 || _deltaT <= 0.0) {
        return -1;
    }
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr) {
        opserr << "WARNING CudaExplicitAlpha::newStep() - no AnalysisModel set\n";
        return -2;
    }
    CudaBcsrLinSOE *cudaSOE = nullptr;
    if (validateCudaSOE(cudaSOE) != 0) {
        return -3;
    }
    const bool deviceOn = cudaSOE->isCudaDeviceEnabled();
    if (deviceOn) {
        ensureDeviceImpl(cudaSOE);
        if (m_impl == nullptr) {
            opserr << "ERROR CudaExplicitAlpha::newStep() - GPU state not initialized; "
                   << "rebuild analysis with `system CuDSS`.\n";
            return -2;
        }
        cudaSOE->setXSyncMode(false);
    } else if (dynamic_cast<DistributedCudaBcsrLinSOE *>(this->getLinearSOE()) == nullptr) {
        opserr << "ERROR CudaExplicitAlpha::newStep() - GPU state not initialized; "
               << "rebuild analysis with `system CuDSS`.\n";
        return -2;
    }

    auto *distSOE = dynamic_cast<DistributedCudaBcsrLinSOE *>(this->getLinearSOE());
    // Drop any leftover B-skip flag from a prior formUnbalance that never reached solve().
    if (distSOE != nullptr)
        distSOE->setGlobalRhsOnDevice(false);

    // P0's integrator vectors only contain locally seeded DOFs after domainChanged.
    // Gather once, then skip on steady EQ steps (P0 already owns global GPU kinematics).
    if (distSOE != nullptr && motionNeedsGather) {
        if (distSOE->gatherOwnedToRoot(*U) < 0 ||
            distSOE->gatherOwnedToRoot(*Udot) < 0 ||
            distSOE->gatherOwnedToRoot(*Udotdot) < 0) {
            opserr << "ERROR CudaExplicitAlpha::newStep() - motion gather to root failed\n";
            return -5;
        }
        motionNeedsGather = false;
        deviceMotionStale = true;  // host motion may have gained remote DOFs
    }

    *Ut = *U;
    *Utdot = *Udot;
    *Utdotdot = *Udotdot;
    if (_deltaT != deltaT) {
        deltaT = _deltaT;
        operatorsBuilt = false;
    }
    if (!operatorsBuilt) {
        if (deviceOn && m_impl == nullptr) {
            return -4;
        }
        // Workers still enter formOperators for collective assembly (deviceOn==false path).
        if (formOperators(cudaSOE) != 0) {
            opserr << "ERROR CudaExplicitAlpha::newStep() - formOperators failed\n";
            return -4;
        }
        operatorsBuilt = true;
    }

    if (initAccel && startPending) {
        if (applyStartCorrection(cudaSOE) != 0) {
            opserr << "ERROR CudaExplicitAlpha::newStep() - starting correction (-initAccel) failed\n";
            return -7;
        }
        startPending = false;
        *Ut = *U;
        *Utdot = *Udot;
        *Utdotdot = *Udotdot;
    }

    if (deviceOn) {
        if (m_impl->newStepPredictor(this) != 0) {
            opserr << "ERROR CudaExplicitAlpha::newStep() - newStepPredictor failed\n";
            return -5;
        }
    }
    // Broadcast predicted kinematics so workers share P0's GPU predictor result.
    // On failure paths above, workers may already be waiting here — keep success path collective.
    if (distSOE != nullptr) {
        if (distSOE->broadcastFromRoot(*U) < 0 ||
            distSOE->broadcastFromRoot(*Udot) < 0 ||
            distSOE->broadcastFromRoot(*Udotdot) < 0 ||
            distSOE->broadcastFromRoot(*Ualpha) < 0 ||
            distSOE->broadcastFromRoot(*Ualphadot) < 0 ||
            distSOE->broadcastFromRoot(*Ualphadotdot) < 0) {
            opserr << "ERROR CudaExplicitAlpha::newStep() - motion broadcast failed\n";
            return -5;
        }
    } else if (!deviceOn) {
        opserr << "ERROR CudaExplicitAlpha::newStep() - GPU device disabled without DistributedCuDSS\n";
        return -5;
    }

    theModel->setResponse(*Ualpha, *Ualphadot, *Ualphadotdot);
    // Domain::update(newTime, dT) ignores dT: it sets ops_Dt = newTime - committedTime = alphaF*dt,
    // while element states were committed at the previous alpha level, a full dt earlier. Materials
    // that integrate over ops_Dt (e.g. ViscousDamper) then run on a slowed clock. Shift the domain's
    // committed time so ops_Dt = dt at the alpha-level evaluation (Domain::commit resets it).
    // (-updateElemDisp commits at t_{n+1}, so ops_Dt = alphaF*dt is already right there.)
    Domain *theDomain = updElemDisp ? 0 : theModel->getDomainPtr();
    if (theDomain != 0) {
        tStepStart = theModel->getCurrentDomainTime();
        theDomain->setCommittedTime(tStepStart + alphaF * deltaT - deltaT);
    }
    if (theModel->updateDomain(theModel->getCurrentDomainTime() + alphaF * deltaT, deltaT) < 0) {
        opserr << "ERROR CudaExplicitAlpha::newStep() - updateDomain failed at t="
               << theModel->getCurrentDomainTime()
               << " (alphaF*dt trial), |Ualpha|=" << Ualpha->Norm()
               << " |Ualphadot|=" << Ualphadot->Norm()
               << " |Ualphadotdot|=" << Ualphadotdot->Norm() << "\n";
        return -6;
    }
    return 0;
}

int CudaExplicitAlpha::formUnbalance()
{
    auto *distSOE = dynamic_cast<DistributedCudaBcsrLinSOE *>(this->getLinearSOE());
    if (distSOE != nullptr)
        distSOE->setGlobalRhsOnDevice(false);

    if (TransientIntegrator::formUnbalance() < 0) {
        return -1;
    }
    CudaBcsrLinSOE *cudaSOE = nullptr;
    if (validateCudaSOE(cudaSOE) != 0) {
        return -2;
    }

    if (distSOE != nullptr) {
        // Merge Phat to root, clear local myB (anti-double-count), then either
        // upload Phat (alphaM~=alphaF) or transform on GPU. Status broadcast keeps
        // every rank's globalRhsOnDevice flag matched so DistCuDSS::solve collectives
        // cannot diverge on a root-only GPU failure.
        int rc = distSOE->mergeBToRoot();
        if (rc == 0)
            distSOE->zeroB();

        Vector status(1);
        if (distSOE->getProcessID() != 0) {
            if (distSOE->broadcastFromRoot(status) < 0)
                return -2;
            if (status(0) < 0.0)
                return -2;
            distSOE->setGlobalRhsOnDevice(true);
            return 0;
        }

        if (rc == 0) {
            if (m_impl == nullptr || !cudaSOE->isCudaDeviceEnabled()) {
                opserr << "ERROR CudaExplicitAlpha::formUnbalance() - rank 0 has no CUDA device\n";
                rc = -2;
            } else if (areAlphaMFClose()) {
                // A = alphaM*M; device B is the merged Phat (no Atilde*M^{-1} transform).
                cudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
                cudaSOE->syncBToDevice();
                rc = 0;
            } else {
                cudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
                rc = m_impl->formUnbalance(cudaSOE);
            }
        }

        status(0) = (rc == 0) ? 1.0 : -1.0;
        if (distSOE->broadcastFromRoot(status) < 0)
            return -2;
        if (rc == 0)
            distSOE->setGlobalRhsOnDevice(true);
        return rc;
    }

    if (areAlphaMFClose()) {
        return 0;  // serial: B already holds Phat; primary SOE has A = alphaM*M
    }
    if (m_impl == nullptr)
        return -1;
    return m_impl->formUnbalance(cudaSOE);
}

int CudaExplicitAlpha::update(const Vector &aiPlusOne)
{
    updateCount++;
    if (updateCount > 1) {
        opserr << "ERROR CudaExplicitAlpha::update() - called " << updateCount
               << " times in one step (Linear algorithm only)\n";
        return -1;  // linear algorithm only
    }
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr || U->Size() <= 0 || aiPlusOne.Size() != U->Size()) {
        return -2;
    }
    CudaBcsrLinSOE *cudaSOE = nullptr;
    if (validateCudaSOE(cudaSOE) != 0) {
        return -4;
    }

    if (cudaSOE->isCudaDeviceEnabled()) {
        if (m_impl == nullptr || m_impl->updateState(cudaSOE, incrementalAccel) != 0) {
            return -4;
        }
    } else {
        // Worker: X already broadcast on host by DistributedCudaBcsrLinSOE::solve.
        if (incrementalAccel) {
            Udotdot->addVector(1.0, aiPlusOne, 1.0);
        } else {
            *Udotdot = aiPlusOne;
        }
    }

    theModel->setVel(*Udot);
    theModel->setAccel(*Udotdot);
    if (theModel->updateDomain() < 0) {
        opserr << "ERROR CudaExplicitAlpha::update() - updateDomain failed"
               << " |U|=" << U->Norm()
               << " |Udot|=" << Udot->Norm()
               << " |Udotdot|=" << Udotdot->Norm()
               << " updateCount=" << updateCount << "\n";
        return -3;
    }
    theModel->setDisp(*U);
    return 0;
}

int CudaExplicitAlpha::commit()
{
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr) {
        return -1;
    }
    // Domain was left at t + alphaF*dt after newStep; advance to t + dt.
    theModel->setCurrentDomainTime(theModel->getCurrentDomainTime() + (1.0 - alphaF) * deltaT);
    if (updElemDisp) {
        theModel->updateDomain();
    }
    return theModel->commitDomain();
}

int CudaExplicitAlpha::revertToLastStep()
{
    // undo the committed-time shift of newStep (Domain::revertToLastCommit used the shifted time)
    if (tStepStart >= 0.0) {
        if (Domain *theDomain = this->getAnalysisModel() ? this->getAnalysisModel()->getDomainPtr() : 0) {
            theDomain->setCurrentTime(tStepStart);
            theDomain->setCommittedTime(tStepStart);
        }
    }
    CudaBcsrLinSOE *cudaSOE = nullptr;
    if (validateCudaSOE(cudaSOE) == 0) {
        cudaSOE->setXSyncMode(true);
    }
    if (U->Size() > 0) {
        *U = *Ut;
        *Udot = *Utdot;
        *Udotdot = *Utdotdot;
    }
    motionNeedsGather = true;
    deviceMotionStale = true;
    return 0;
}

double CudaExplicitAlpha::getCFactor() { return c2; }

const Vector &CudaExplicitAlpha::getVel() { return *Ualphadot; }  // alpha-level velocity

int CudaExplicitAlpha::sendSelf(int cTag, Channel &theChannel)
{
    Vector data(8);
    data(0) = alphaM;
    data(1) = alphaF;
    data(2) = beta;
    data(3) = gamma;
    data(4) = updElemDisp ? 1.0 : 0.0;
    data(5) = incrementalAccel ? 1.0 : 0.0;
    data(6) = useAlphaCloseCheck ? 1.0 : 0.0;
    data(7) = initAccel ? 1.0 : 0.0;
    return theChannel.sendVector(this->getDbTag(), cTag, data);
}

int CudaExplicitAlpha::recvSelf(int cTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    Vector data(8);
    if (theChannel.recvVector(this->getDbTag(), cTag, data) < 0) {
        return -1;
    }
    alphaM = data(0);
    alphaF = data(1);
    beta = data(2);
    gamma = data(3);
    updElemDisp = data(4) > 0.5;
    incrementalAccel = data(5) > 0.5;
    useAlphaCloseCheck = data(6) > 0.5;
    initAccel = data(7) > 0.5;
    operatorsBuilt = false;
    return 0;
}

void CudaExplicitAlpha::Print(OPS_Stream &s, int flag)
{
    (void)flag;
    s << "CudaExplicitAlpha alphaF=" << alphaF << " alphaM=" << alphaM
      << " initAccel=" << (initAccel ? 1 : 0);
    s << endln;
}

int CudaExplicitAlpha::revertToStart()
{
    CudaBcsrLinSOE *cudaSOE = nullptr;
    if (validateCudaSOE(cudaSOE) == 0) {
        cudaSOE->setXSyncMode(true);
    }
    if (U->Size() > 0) {
        Ut->Zero();
        Utdot->Zero();
        Utdotdot->Zero();
        U->Zero();
        Udot->Zero();
        Udotdot->Zero();
        if (m_impl != nullptr && m_impl->getSize() > 0) {
            m_impl->zeroState();
        }
    }
    startPending = true;
    deviceMotionStale = true;
    return 0;
}
