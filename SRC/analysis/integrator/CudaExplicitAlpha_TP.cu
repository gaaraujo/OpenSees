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

// Developed: Chinmoy Kolay (chk311@lehigh.edu)
// Implemented: Andreas Schellenberg (andreas.schellenberg@gmail.com)
// CUDA implementation: Gustavo A. Araujo R. (garaujor@stanford.edu)
//
// Description: GPU explicit Kolay-Ricles trapezoidal-rule (TP) integrator for
// CudaBcsrLinSOE + cuDSS, based on the CPU ExplicitAlpha_TP family.
//
// Reference: Kolay, C. and J. Ricles (2014). "Development of a family of
// unconditionally stable explicit direct integration algorithms with
// controllable numerical energy dissipation." Earthquake Engineering and
// Structural Dynamics, 43(9):1361-1380.

#include <CudaExplicitAlpha_TP.h>
#include <CudaBcsrLinSOE.h>
#include <DistributedCudaBcsrLinSOE.h>
#include <AnalysisModel.h>
#include <Channel.h>
#include <DOF_Group.h>
#include <DOF_GrpIter.h>
#include <FE_Element.h>
#include <FEM_ObjectBroker.h>
#include <LinearSOE.h>
#include <Vector.h>
#include <classTags.h>
#include <elementAPI.h>
#include <cmath>
#include <cstring>

#include "CudaCsrMatrix.h"
#include "CudaUtils.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/memory.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <vector>

using thrust::raw_pointer_cast;

using namespace CudaUtils;


// Kolay-Ricles trapezoidal-rule (TP) integrator on GPU (CudaBcsrLinSOE + cuDSS).
// Unlike the midpoint CUDA path, TP uses two residual passes per step and a blended Put vector.
// Device scalar type T follows SOE precision (dDDI=double, dFFI=float); host stays double.

namespace {

// --- Templated device kernels ---

template<typename T>
__global__ void kernelAxpby(int n, T alpha, const T *x, T beta, T *y)
{
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

// TP displacement/velocity predictor + trial acceleration (full-step kinematics, not alpha-interpolated).
// alphaSol1 = alpha^{-1} M Uddot_n; alphaSol3 = alpha_3 Uddot_n when alphaM != alphaF.
template<typename T>
__global__ void kernelNewStepTP(int n, T dt, T gamma, T alphaF, T alphaM, const T *alphaSol1,
                                const T *alphaSol3, const T *Ut, const T *Utdot, const T *Utdotdot, T *U, T *Udot,
                                T *Uddot, int incrementalAccel, int alphaClose)
{
    const int stride = blockDim.x * gridDim.x;
    const T invAlphaF = static_cast<T>(1.0) / alphaF;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const T uhat = dt * alphaSol1[i];  // Udot_hat = dt * alpha_1 * Uddot_n
        U[i] = Ut[i] + dt * Utdot[i] + (static_cast<T>(0.5) + gamma) * dt * uhat;
        Udot[i] = Utdot[i] + uhat;
        // Trial accel for the second residual pass at t_{n+1}
        if (incrementalAccel) {
            Uddot[i] = Utdotdot[i] * invAlphaF;
        } else if (alphaClose) {
            Uddot[i] = (static_cast<T>(1.0) - alphaM) * Utdotdot[i] * invAlphaF;
        } else {
            Uddot[i] = (Utdotdot[i] - alphaSol3[i]) * invAlphaF;
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
struct CudaExplicitAlpha_TP::ImplBase {
    struct HostMotionPtrs {
        double *U;
        double *Udot;
        double *Uddot;
        double *Ut;       // step-start backup (maps to h_state_prev)
        double *Utdot;
        double *Utdotdot;
    };

    virtual ~ImplBase() = default;

    virtual void allocate(int n, int alphaNumRhs) = 0;
    virtual void destroySolvers() = 0;
    virtual void shutdown() = 0;
    virtual void ensureAlphaBuffers(int alphaNumRhs) = 0;
    virtual int formOperators(CudaExplicitAlpha_TP *integrator, CudaBcsrLinSOE *cudaSOE) = 0;
    virtual HostMotionPtrs ensureHostMotionBuffers(int n) = 0;
    virtual int newStepPredictor(CudaExplicitAlpha_TP *integrator) = 0;
    virtual int formUnbalanceFromPut(CudaBcsrLinSOE *cudaSOE, bool alphaClose) = 0;
    virtual int updateState(CudaBcsrLinSOE *cudaSOE, bool incrementalAccel, double alphaF) = 0;
    virtual void zeroState() = 0;
    virtual int getSize() const = 0;
    virtual double *getPutHostPtr() = 0;
    virtual int capturePutFromDeviceB(CudaBcsrLinSOE *cudaSOE) = 0;
    virtual void blendPutFromHostB(CudaBcsrLinSOE *cudaSOE, double oneMinusAlphaF, double alphaF) = 0;
    // -order2
    virtual int order2PreStep(CudaExplicitAlpha_TP *integrator, double *hostDv) = 0;
    virtual int addRateShiftPut() = 0;
    virtual int applyStart(CudaExplicitAlpha_TP *integrator, CudaBcsrLinSOE *cudaSOE, double *hostA1) = 0;
    virtual int alpha1Accel(double *hostOut) = 0;
};

template<typename T>
struct ImplT_TP : CudaExplicitAlpha_TP::ImplBase {
    int size = 0;
    int alphaNumRhs = 2;  // 1 when alphaM ~= alphaF, or when incrementalAccel (alpha_3 unused)

    thrust::device_vector<T> d_state_cur;    // [U | Udot | Uddot]
    thrust::device_vector<T> d_state_prev;   // [Ut | Utdot | Utdotdot] step-start backup
    thrust::device_vector<T> d_w2;           // workspace for M^{-1} and solves
    thrust::device_vector<T> d_put;          // blended TP unbalance on device
    thrust::device_vector<T> d_alphaRhs;     // RHS for alpha-operator solve(s)
    thrust::device_vector<T> d_alphaSol;     // alphaSol1 [| alphaSol3]
    thrust::device_vector<T> d_dv;           // -order2 velocity shift dv
    thrust::device_vector<T> d_w3;           // -order2 C_int product
    bool useMatC = false;                    // -order2: C_int operator built
    bool alpha1Ready = false;                // alphaSol1 holds alpha1*Uddot_n for this step's predictor
    pinned_host_vector<double> h_state_cur;
    pinned_host_vector<double> h_state_prev;
    pinned_host_vector<double> h_put;          // mirrored by OpenSees Vector *Put

    cusparseHandle_t cusparseHandle = nullptr;
    cudaStream_t stream = nullptr;           // cached SOE stream from getCudaStream()
    CudaCsrMatrix *matM = nullptr;             // mass operator M
    CudaCsrMatrix *matAlpha = nullptr;         // alpha = M + gamma*dt*C + beta*dt^2*K
    CudaCsrMatrix *matA = nullptr;             // A = alphaM*M + alphaF*gamma*dt*C + alphaF*beta*dt^2*K
    CudaCsrMatrix *matC = nullptr;             // C_int (-order2; spmv only)
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
    void allocate(int n, int alphaNumRhsIn) override
    {
        size = n;
        alphaNumRhs = alphaNumRhsIn;
        const T zero = static_cast<T>(0.0);
        const std::size_t motionSize = static_cast<std::size_t>(kMotionFields * n);
        d_state_cur.assign(motionSize, zero);
        d_state_prev.assign(motionSize, zero);
        d_w2.assign(n, zero);
        d_put.assign(n, zero);
        d_alphaRhs.assign(alphaNumRhs * n, zero);
        d_alphaSol.assign(alphaNumRhs * n, zero);
        d_dv.assign(n, zero);
        d_w3.assign(n, zero);
        alpha1Ready = false;
        h_put.assign(static_cast<std::size_t>(n), 0.0);
    }

    HostMotionPtrs ensureHostMotionBuffers(int n) override
    {
        HostMotionPtrs buf{};
        if (n <= 0) {
            return buf;
        }
        const std::size_t motionSize = static_cast<std::size_t>(kMotionFields) * static_cast<std::size_t>(n);
        h_state_cur.resize(motionSize);
        h_state_prev.resize(motionSize);
        h_put.resize(static_cast<std::size_t>(n));
        double *cur = raw_pointer_cast(h_state_cur.data());
        double *prev = raw_pointer_cast(h_state_prev.data());
        buf.U = cur;
        buf.Udot = cur + n;
        buf.Uddot = cur + 2 * n;
        buf.Ut = prev;
        buf.Utdot = prev + n;
        buf.Utdotdot = prev + 2 * n;
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
        if (matC != nullptr) {
            matC->detachSharedSpmvPattern();
        }
        delete matAlpha;
        delete matA;
        delete matC;
        delete matM;
        matM = nullptr;
        matAlpha = nullptr;
        matA = nullptr;
        matC = nullptr;
        useMatC = false;
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
        if (stream == nullptr) {
            return -1;
        }
        if (cusparseHandle != nullptr) {
            cuSparseCheckError(cusparseSetStream(cusparseHandle, stream), "cusparseSetStream");
        }
        return 0;
    }

    void initCusparse()
    {
        if (cusparseHandle == nullptr) {
            cuSparseCheckError(cusparseCreate(&cusparseHandle), "cusparseCreate");
        }
        if (stream != nullptr) {
            cuSparseCheckError(cusparseSetStream(cusparseHandle, stream), "cusparseSetStream");
        }
    }

    void ensureMatrices(CudaPrecision prec, bool withC)
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

            if (withC) {  // C_int, -order2 only
                makeMatrixConfigs(prec, solver, spmv, exec, matM->getSpmvBackend());
                matC = new CudaCsrMatrix(solver, spmv, exec);
            }
        }
    }

    int bindSharedStructure(CudaBcsrLinSOE *soe)
    {
        // Reuse one CSR pattern for matM, matAlpha, and matA (values differ).
        const int nnz = soe->getNumNonZeroValues();
        const int *rowPtr = soe->getDeviceRowPtrs();
        const int *colIdx = soe->getDeviceColIndices();
        if (rowPtr == nullptr || colIdx == nullptr) {
            return -1;
        }

        const bool samePattern = boundStructureRows == size && boundStructureNnz == nnz && boundRowPtr == rowPtr &&
                                 boundColIdx == colIdx && matM != nullptr && matM->isStructureBound();

        if (!samePattern) {
            if (matM != nullptr && matM->isStructureBound()) {
                matM->reset();
                matAlpha->reset();
                matA->reset();
                if (matC != nullptr)
                    matC->reset();
            }
            if (matM->bindStructure(size, nnz, rowPtr, colIdx) != 0) {
                return -1;
            }
            if (matAlpha->bindStructure(size, nnz, rowPtr, colIdx) != 0) {
                return -1;
            }
            if (matA->bindStructure(size, nnz, rowPtr, colIdx) != 0) {
                return -1;
            }
            if (matC != nullptr && matC->bindStructure(size, nnz, rowPtr, colIdx) != 0) {
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

    // Build/refactor GPU operators used by newStepPredictor() and formUnbalanceFromPut().
    // Finish all collective assemble/merges before GPU factorize (see CudaExplicitAlpha).
    int formOperators(CudaExplicitAlpha_TP *integrator, CudaBcsrLinSOE *cudaSOE) override
    {
        const double bdt2 = integrator->beta * integrator->deltaT * integrator->deltaT;
        const double gdt = integrator->gamma * integrator->deltaT;
        const bool areClose = integrator->areAlphaMFClose();
        // alpha_3 column is only consumed by the non-incremental predictor kernel.
        // Match ExplicitAlpha_TP::newStep, which skips alpha3 under -incrementalAccel.
        const int numRhs = (areClose || integrator->incrementalAccel) ? 1 : 2;
        ensureAlphaBuffers(numRhs);

        if (bindStream(cudaSOE) != 0) {
            return -1;
        }
        cudaSOE->syncIndicesToDevice();
        cudaSOE->ensureDeviceVectorSizes();
        ensureMatrices(cudaSOE->getPrecision(), integrator->order2);
        if (bindSharedStructure(cudaSOE) != 0) {
            return -1;
        }

        void *rhsM = thrust::raw_pointer_cast(d_alphaRhs.data());
        void *solM = thrust::raw_pointer_cast(d_w2.data());
        void *rhsAlpha = rhsM;
        void *solAlpha1 = thrust::raw_pointer_cast(d_alphaSol.data());

        // --- Mass operator M ---
        if (integrator->formTangentIntoSOE(INITIAL_TANGENT, 0.0, 0.0, 1.0) != 0) {
            return -1;
        }
        cudaSOE->syncAValuesToDevice();
        if (matM->copyValues(cudaSOE->getDeviceAValues()) != 0) {
            return -2;
        }

        // --- alpha (snapshot only; factorize after all merges) ---
        if (integrator->formTangentIntoSOE(INITIAL_TANGENT, bdt2, gdt, 1.0) != 0) {
            return -3;
        }
        cudaSOE->syncAValuesToDevice();
        if (matAlpha->copyValues(cudaSOE->getDeviceAValues()) != 0) {
            return -4;
        }

        // --- C_int (-order2; before A so the primary SOE still holds A for the solve) ---
        if (integrator->order2) {
            if (integrator->formTangentIntoSOE(INITIAL_TANGENT, 0.0, 1.0, 0.0) != 0) {
                return -7;
            }
            cudaSOE->syncAValuesToDevice();
            if (matC->copyValues(cudaSOE->getDeviceAValues()) != 0) {
                return -7;
            }
            useMatC = true;
        }
        alpha1Ready = false;

        // --- A / Ã ---
        if (areClose) {
            if (integrator->formTangentIntoSOE(INITIAL_TANGENT, 0.0, 0.0, integrator->alphaM) != 0) {
                return -5;
            }
        } else if (integrator->formTangentIntoSOE(INITIAL_TANGENT, integrator->alphaF * bdt2,
                                                   integrator->alphaF * gdt, integrator->alphaM) != 0) {
            return -5;
        }
        cudaSOE->syncAValuesToDevice();
        if (matA->copyValues(cudaSOE->getDeviceAValues()) != 0) {
            opserr << "ERROR CudaExplicitAlpha_TP::formOperators - matA->copyValues failed\n";
            return -5;
        }

        // --- GPU factorize AFTER all MPI merges are complete ---
        if (!matAlpha->isFactored()) {
            if (matAlpha->factorize(rhsAlpha, solAlpha1, numRhs, nullptr) != 0) {
                return -4;
            }
        } else if (matAlpha->refactorize(rhsAlpha, solAlpha1, numRhs) != 0) {
            return -4;
        }
        if (!matM->isFactored()) {
            if (matM->factorize(rhsM, solM, 1, matAlpha) != 0) {
                return -2;
            }
        } else if (matM->refactorize(rhsM, solM) != 0) {
            return -2;
        }
        return 0;
    }

    // GPU predictor: alpha_1*Uddot_n solve, then full-step trial (U, Udot, Uddot).
    int newStepPredictor(CudaExplicitAlpha_TP *integrator) override
    {
        if (size > 0) {
            if (integrator->deviceMotionStale) {
                d_state_cur = h_state_cur;
                d_state_prev = h_state_prev;
                integrator->deviceMotionStale = false;
            } else {
                // Committed state already on device from last updateState; step-start
                // backup is that same state (host *Ut=*U already mirrored).
                d_state_prev = d_state_cur;
            }
        }

        const T *dUt = raw_pointer_cast(d_state_prev.data());
        const T *dUtdot = dUt + size;
        const T *dUtdotdot = dUt + 2 * size;

        T *dU = devU();
        T *dUdot = devUdot();
        T *dUddot = devUddot();
        T *dAlphaSol1 = thrust::raw_pointer_cast(d_alphaSol.data());
        T *dAlphaSol3 = alphaNumRhs > 1 ? dAlphaSol1 + size : nullptr;
        T *dAlphaRhs = thrust::raw_pointer_cast(d_alphaRhs.data());

        const T dt = static_cast<T>(integrator->deltaT);
        const T gammaT = static_cast<T>(integrator->gamma);
        const T alphaFT = static_cast<T>(integrator->alphaF);
        const T alphaMT = static_cast<T>(integrator->alphaM);
        const bool alphaClose = integrator->areAlphaMFClose();
        const int incAccel = integrator->incrementalAccel ? 1 : 0;
        // Need alpha_3 = I - alpha^{-1} A only for non-incremental trial accel when alphaM != alphaF.
        const bool needAlpha3 = (incAccel == 0) && !alphaClose;

        // -order2: alphaSol1 already holds alpha1*Uddot_n (order2PreStep); alpha_3 still needs both columns.
        const bool reuseAlpha1 = alpha1Ready && !needAlpha3;
        alpha1Ready = false;
        if (!reuseAlpha1)
            applyM(dUtdotdot, dAlphaRhs);
        if (needAlpha3) {
            matA->spmv(dUtdotdot, dAlphaRhs + size);
            if (matAlpha->solve(dAlphaRhs, dAlphaSol1, alphaNumRhs) != 0) {
                return -5;
            }
            kernelNewStepTP<<<gridBlocks(size), 256, 0, stream>>>(size, dt, gammaT, alphaFT, alphaMT, dAlphaSol1,
                                                                  dAlphaSol3, dUt, dUtdot, dUtdotdot, dU, dUdot, dUddot,
                                                                  incAccel, 0);
        } else {
            if (!reuseAlpha1 && matAlpha->solve(dAlphaRhs, dAlphaSol1, 1) != 0) {
                return -5;
            }
            kernelNewStepTP<<<gridBlocks(size), 256, 0, stream>>>(
                size, dt, gammaT, alphaFT, alphaMT, dAlphaSol1, static_cast<const T *>(nullptr), dUt, dUtdot,
                dUtdotdot, dU, dUdot, dUddot, incAccel, alphaClose ? 1 : 0);
        }
        cudaCheckError(cudaStreamSynchronize(stream), "integrator stream sync before host read");
        h_state_cur = d_state_cur;
        return 0;
    }

    // alphaSol1 <- alpha1 * x (second column zeroed when two RHS are factorized)
    int solveAlpha1(const T *x)
    {
        T *dAlphaRhs = thrust::raw_pointer_cast(d_alphaRhs.data());
        T *dAlphaSol1 = thrust::raw_pointer_cast(d_alphaSol.data());
        if (alphaNumRhs > 1) {
            thrust::fill(d_alphaRhs.begin() + size, d_alphaRhs.end(), static_cast<T>(0.0));
        }
        applyM(x, dAlphaRhs);
        return matAlpha->solve(dAlphaRhs, dAlphaSol1, alphaNumRhs);
    }

    void copyToHost(const thrust::device_vector<T> &d, double *hostOut)
    {
        std::vector<T> tmp(static_cast<std::size_t>(size));
        thrust::copy(d.begin(), d.begin() + size, tmp.begin());
        for (int i = 0; i < size; ++i)
            hostOut[i] = static_cast<double>(tmp[i]);
    }

    // -order2, before the n-end pass: dv = gamma*dt*alpha1*a_n, on device and host.
    int order2PreStep(CudaExplicitAlpha_TP *integrator, double *hostDv) override
    {
        if (size > 0 && integrator->deviceMotionStale) {
            d_state_cur = h_state_cur;
            d_state_prev = h_state_prev;
            integrator->deviceMotionStale = false;
            alpha1Ready = false;
        }
        if (!alpha1Ready && solveAlpha1(devUddot()) != 0) {
            return -1;
        }
        alpha1Ready = true;
        const T *x = thrust::raw_pointer_cast(d_alphaSol.data());
        T *dv = thrust::raw_pointer_cast(d_dv.data());
        cudaMemsetAsync(dv, 0, static_cast<std::size_t>(size) * sizeof(T), stream);
        kernelAxpby<<<gridBlocks(size), 256, 0, stream>>>(
            size, static_cast<T>(integrator->gamma * integrator->deltaT), x, static_cast<T>(0.0), dv);
        cudaCheckError(cudaStreamSynchronize(stream), "integrator stream sync after -order2 dv");
        copyToHost(d_dv, hostDv);
        if (integrator->velRecoverPending) {
            // after a mid-analysis domainChanged: v_alg = v_rep - dv on the host mirrors, then re-upload
            for (int i = 0; i < size; ++i) {
                (*integrator->Udot)(i) -= hostDv[i];
                (*integrator->Utdot)(i) -= hostDv[i];
            }
            d_state_cur = h_state_cur;
            d_state_prev = h_state_prev;
            integrator->velRecoverPending = false;
        }
        return 0;
    }

    // -order2: Put += C_int * dv (after the blend), host mirror refreshed.
    int addRateShiftPut() override
    {
        if (!useMatC) {
            return -7;
        }
        T *w3 = thrust::raw_pointer_cast(d_w3.data());
        if (matC->spmv(thrust::raw_pointer_cast(d_dv.data()), w3) != 0) {
            return -7;
        }
        kernelAxpby<<<gridBlocks(size), 256, 0, stream>>>(size, static_cast<T>(1.0), static_cast<const T *>(w3),
                                                            static_cast<T>(1.0),
                                                            thrust::raw_pointer_cast(d_put.data()));
        cudaCheckError(cudaStreamSynchronize(stream), "integrator stream sync after -order2 Put shift");
        h_put = d_put;
        return 0;
    }

    // -initAccel / -order2 start: host B holds R0 = p - C v0 - f(u0, v0); a0 = M^{-1} R0 and, with
    // -order2, Udot -= gamma*dt*alpha1*a0.
    int applyStart(CudaExplicitAlpha_TP *integrator, CudaBcsrLinSOE *cudaSOE, double *hostA1) override
    {
        if (bindStream(cudaSOE) != 0 || matM == nullptr || !matM->isFactored()) {
            return -1;
        }
        if (size > 0 && integrator->deviceMotionStale) {
            d_state_cur = h_state_cur;
            d_state_prev = h_state_prev;
            integrator->deviceMotionStale = false;
        }
        cudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
        cudaSOE->syncBToDevice();
        if (matM->solve(cudaSOE->getDeviceB(), devUddot()) != 0) {
            return -2;
        }
        if (!integrator->order2) {
            cudaCheckError(cudaStreamSynchronize(stream), "integrator stream sync after -initAccel start");
            h_state_cur = d_state_cur;
            alpha1Ready = false;
            return 0;
        }
        if (solveAlpha1(devUddot()) != 0) {
            return -3;
        }
        kernelAxpby<<<gridBlocks(size), 256, 0, stream>>>(
            size, static_cast<T>(-integrator->gamma * integrator->deltaT),
            static_cast<const T *>(thrust::raw_pointer_cast(d_alphaSol.data())), static_cast<T>(1.0), devUdot());
        cudaCheckError(cudaStreamSynchronize(stream), "integrator stream sync after -order2 start");
        h_state_cur = d_state_cur;
        copyToHost(d_alphaSol, hostA1);
        alpha1Ready = true;  // alphaSol1 = alpha1*a0, the first step's
        return 0;
    }

    // -order2 reporting: alpha1 * Uddot_{n+1} (cached for the next step) to the host.
    int alpha1Accel(double *hostOut) override
    {
        if (solveAlpha1(devUddot()) != 0) {
            return -1;
        }
        cudaCheckError(cudaStreamSynchronize(stream), "integrator stream sync after -order2 alpha1 accel");
        copyToHost(d_alphaSol, hostOut);
        alpha1Ready = true;
        return 0;
    }

    double *getPutHostPtr() override { return raw_pointer_cast(h_put.data()); }

    // Put <- B^(1): sync SOE B to device and copy into d_put (B is overwritten in pass 2).
    int capturePutFromDeviceB(CudaBcsrLinSOE *cudaSOE) override
    {
        if (bindStream(cudaSOE) != 0) {
            return -1;
        }
        cudaSOE->syncBToDevice();
        const T *dB = static_cast<const T *>(cudaSOE->getDeviceB());
        if (dB == nullptr) {
            return -2;
        }
        T *dPut = thrust::raw_pointer_cast(d_put.data());
        cudaCheckError(cudaMemcpyAsync(dPut, dB, static_cast<std::size_t>(size) * sizeof(T),
                                       cudaMemcpyDeviceToDevice, stream),
                       "capture Put from device B");
        cudaCheckError(cudaStreamSynchronize(stream), "integrator stream sync after Put capture");
        return 0;
    }

    void blendPutFromHostB(CudaBcsrLinSOE *cudaSOE, double oneMinusAlphaF, double alphaF) override
    {
        if (bindStream(cudaSOE) != 0) {
            return;
        }
        cudaSOE->syncBToDevice();
        const T *dB = static_cast<const T *>(cudaSOE->getDeviceB());
        T *dPut = thrust::raw_pointer_cast(d_put.data());
        // Put <- (1 - alphaF) * Put + alphaF * B^(2)
        kernelAxpby<<<gridBlocks(size), 256, 0, stream>>>(size, static_cast<T>(alphaF), dB,
                                                            static_cast<T>(oneMinusAlphaF), dPut);
        cudaCheckError(cudaStreamSynchronize(stream), "integrator stream sync after Put blend");
        h_put = d_put;
    }

    // CudaExplicitAlpha_TP::formUnbalance() (from Linear::solveCurrentStep): B <- Put or B <- alpha * M^{-1} * Put.
    int formUnbalanceFromPut(CudaBcsrLinSOE *cudaSOE, bool alphaClose) override
    {
        if (bindStream(cudaSOE) != 0) {
            return -1;
        }
        if (alphaClose) {
            // Proportional shortcut: primary SOE holds A = alphaM*M, so B = Put directly.
            return cudaSOE->setDeviceB(thrust::raw_pointer_cast(d_put.data()), size);
        }

        T *w2 = thrust::raw_pointer_cast(d_w2.data());
        void *put = thrust::raw_pointer_cast(d_put.data());
        if (matM == nullptr || !matM->isFactored()) {
            return -3;
        }
        if (matM->solve(put, w2) != 0) {
            return -3;
        }
        void *b = cudaSOE->getDeviceB();
        matAlpha->spmv(w2, b);  // B = alpha * M^{-1} * Put
        cudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Device);
        return 0;
    }

    // TP update only refreshes acceleration; U and Udot were set in newStep().
    int updateState(CudaBcsrLinSOE *cudaSOE, bool incrementalAccel, double alphaF) override
    {
        if (bindStream(cudaSOE) != 0) {
            return -1;
        }
        T *dUddot = devUddot();
        const T *dX = static_cast<const T *>(cudaSOE->getDeviceX());
        if (dX == nullptr) {
            return -4;
        }
        if (incrementalAccel) {
            // Match ExplicitAlpha_TP::update addVector(alphaF, aiPlusOne, 1.0):
            // Uddot <- alphaF * Uddot_trial + X  (trial Uddot was set in newStepPredictor).
            const T alphaFT = static_cast<T>(alphaF);
            const T one = static_cast<T>(1.0);
            kernelAxpby<<<gridBlocks(size), 256, 0, stream>>>(size, one, dX, alphaFT, dUddot);
        } else {
            cudaMemcpyAsync(dUddot, dX, static_cast<std::size_t>(size) * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        }
        cudaCheckError(cudaStreamSynchronize(stream), "integrator stream sync before host read");

        double *hUddot = raw_pointer_cast(h_state_cur.data()) + 2 * size;
        if (incrementalAccel) {
            // Pull only accel so host matches GPU accumulation (incl. float SOE).
            thrust::copy(d_state_cur.begin() + 2 * size, d_state_cur.begin() + 3 * size,
                         h_state_cur.begin() + 2 * size);
        } else {
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
        d_put.assign(static_cast<std::size_t>(size), zero);
        alpha1Ready = false;
        std::fill(h_put.begin(), h_put.end(), 0.0);
    }
};

template struct ImplT_TP<double>;
template struct ImplT_TP<float>;

void CudaExplicitAlpha_TP::destroyDeviceImpl()
{
    if (m_impl != nullptr) {
        m_impl->shutdown();
        delete m_impl;
        m_impl = nullptr;
    }
}

void CudaExplicitAlpha_TP::ensureDeviceImpl(CudaBcsrLinSOE *cudaSOE)
{
    // Pick ImplT_TP<float> or ImplT_TP<double> to match uniform SOE precision.
    const CudaPrecision prec = cudaSOE->getPrecision();
    const bool useFloat = (prec == CudaPrecision::dFFI);
    if (m_impl != nullptr && m_deviceUsesFloat == useFloat) {
        return;
    }
    destroyDeviceImpl();
    if (useFloat) {
        m_impl = new ImplT_TP<float>();
    } else {
        m_impl = new ImplT_TP<double>();
    }
    m_deviceUsesFloat = useFloat;
}

int CudaExplicitAlpha_TP::formOperators(CudaBcsrLinSOE *cudaSOE)
{
    const double bdt2 = beta * deltaT * deltaT;
    const double gdt = gamma * deltaT;
    const bool areClose = areAlphaMFClose();
    auto *distSOE = dynamic_cast<DistributedCudaBcsrLinSOE *>(this->getLinearSOE());

    if (!cudaSOE->isCudaDeviceEnabled()) {
        if (formTangentIntoSOE(INITIAL_TANGENT, 0.0, 0.0, 1.0) != 0)
            return -1;
        if (formTangentIntoSOE(INITIAL_TANGENT, bdt2, gdt, 1.0) != 0)
            return -3;
        if (order2 && formTangentIntoSOE(INITIAL_TANGENT, 0.0, 1.0, 0.0) != 0)
            return -7;
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
                opserr << "ERROR CudaExplicitAlpha_TP::formOperators - rank 0 GPU operator build failed\n";
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

bool validateCudaExplicitAlphaTPParams(double alphaF, double alphaM, double gamma, double beta)
{
    if (alphaF < 0.5 || alphaF > 1.0) {
        opserr << "WARNING - invalid alphaF for CudaExplicitAlpha_TP, want 0.5 <= alphaF <= 1.0\n";
        return false;
    }
    if (gamma <= 0.0 || beta <= 0.0) {
        opserr << "WARNING - invalid gamma/beta for CudaExplicitAlpha_TP, want gamma > 0 and beta > 0\n";
        return false;
    }
    if (alphaM < 0.5 || alphaM > 2.0) {
        opserr << "WARNING - recommended for unconditional stability (linear): 0.5 <= alphaM <= 2.0\n";
    }
    if (gamma < 0.5) {
        opserr << "WARNING - recommended for unconditional stability (linear): gamma >= 0.5\n";
    }
    if (beta < 0.5 * gamma) {
        opserr << "WARNING - recommended for unconditional stability (linear): beta >= gamma/2\n";
    }
    return true;
}

bool parseCudaExplicitAlphaTPOptions(CudaExplicitAlpha_TP::Options &opts)
{
    opts = CudaExplicitAlpha_TP::Options{};
    while (OPS_GetNumRemainingInputArgs() > 0) {
        const char *tok = OPS_GetString();
        if (tok == nullptr) {
            break;
        }
        if (strcmp(tok, "-incrementalAccel") == 0) {
            opts.incrementalAccel = true;
        } else if (strcmp(tok, "-alphaCloseCheck") == 0) {
            opts.useAlphaCloseCheck = true;
        } else if (strcmp(tok, "-order2") == 0) {
            opts.order2 = true;
        } else if (strcmp(tok, "-initAccel") == 0) {
            opts.initAccel = true;
        } else {
            opserr << "WARNING CudaExplicitAlpha_TP family - unknown flag " << tok
                   << "; want <-incrementalAccel> <-alphaCloseCheck> <-order2> <-initAccel>\n";
            return false;
        }
    }
    return true;
}

} // namespace

void *OPS_CudaExplicitAlpha_TP(void)
{
    if (OPS_GetNumRemainingInputArgs() < 4) {
        opserr << "WARNING integrator CudaExplicitAlpha_TP alphaF alphaM gamma beta "
                  "<-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }
    double alphaF = 0.0;
    double alphaM = 0.0;
    double gamma = 0.0;
    double beta = 0.0;
    int numData = 1;
    if (OPS_GetDoubleInput(&numData, &alphaF) != 0 || OPS_GetDoubleInput(&numData, &alphaM) != 0 ||
        OPS_GetDoubleInput(&numData, &gamma) != 0 || OPS_GetDoubleInput(&numData, &beta) != 0) {
        opserr << "WARNING CudaExplicitAlpha_TP - invalid alphaF/alphaM/gamma/beta\n";
        return nullptr;
    }
    if (!validateCudaExplicitAlphaTPParams(alphaF, alphaM, gamma, beta)) {
        return nullptr;
    }
    CudaExplicitAlpha_TP::Options opts;
    if (!parseCudaExplicitAlphaTPOptions(opts)) {
        return nullptr;
    }
    return new CudaExplicitAlpha_TP(alphaF, alphaM, gamma, beta, opts);
}

void *OPS_CudaKRAlpha_TP(void)
{
    if (OPS_GetNumRemainingInputArgs() < 1) {
        opserr << "WARNING integrator CudaKRAlpha_TP rhoInf "
                  "<-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }
    double rhoInf = 0.0;
    int numData = 1;
    if (OPS_GetDoubleInput(&numData, &rhoInf) != 0) {
        opserr << "WARNING CudaKRAlpha_TP - invalid rhoInf\n";
        return nullptr;
    }
    const double den = 1.0 + rhoInf;
    const double alphaF = 1.0 / den;
    const double alphaM = (2.0 - rhoInf) / den;
    const double gamma = 0.5 * (3.0 - rhoInf) / den;
    const double beta = 1.0 / (den * den);
    CudaExplicitAlpha_TP::Options opts;
    if (!parseCudaExplicitAlphaTPOptions(opts)) {
        return nullptr;
    }
    return new CudaExplicitAlpha_TP(alphaF, alphaM, gamma, beta, opts);
}

void *OPS_CudaMKRAlpha_TP(void)
{
    if (OPS_GetNumRemainingInputArgs() < 1) {
        opserr << "WARNING integrator CudaMKRAlpha_TP rhoInfEquivalent "
                  "<-incrementalAccel> <-alphaCloseCheck>\n";
        return nullptr;
    }
    double rhoInfEquivalent = 0.0;
    int numData = 1;
    if (OPS_GetDoubleInput(&numData, &rhoInfEquivalent) != 0) {
        opserr << "WARNING CudaMKRAlpha_TP - invalid rhoInfEquivalent\n";
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
    CudaExplicitAlpha_TP::Options opts;
    if (!parseCudaExplicitAlphaTPOptions(opts)) {
        return nullptr;
    }
    return new CudaExplicitAlpha_TP(alphaF, alphaM, gamma, beta, opts);
}

CudaExplicitAlpha_TP::CudaExplicitAlpha_TP()
    : CudaExplicitAlpha_TP(INTEGRATOR_TAGS_CudaExplicitAlpha_TP, 0.5, 0.5, 0.5, 0.25, Options{})
{
}

CudaExplicitAlpha_TP::CudaExplicitAlpha_TP(double _alphaF, double _alphaM, double _gamma, double _beta)
    : CudaExplicitAlpha_TP(_alphaF, _alphaM, _gamma, _beta, Options{})
{
}

CudaExplicitAlpha_TP::CudaExplicitAlpha_TP(double _alphaF, double _alphaM, double _gamma, double _beta,
                                           const Options &opts)
    : CudaExplicitAlpha_TP(INTEGRATOR_TAGS_CudaExplicitAlpha_TP, _alphaF, _alphaM, _gamma, _beta, opts)
{
}

CudaExplicitAlpha_TP::CudaExplicitAlpha_TP(int classTag, double _alphaF, double _alphaM, double _gamma, double _beta,
                                             const Options &opts)
    : TransientIntegrator(classTag),
      alphaM(_alphaM),
      alphaF(_alphaF),
      beta(_beta),
      gamma(_gamma),
      incrementalAccel(opts.incrementalAccel),
      useAlphaCloseCheck(opts.useAlphaCloseCheck),
      order2(opts.order2),
      initAccel(opts.initAccel || opts.order2),
      startPending(true),
      Vrep(new Vector(0)),
      Vforce(new Vector(0)),
      dVres(new Vector(0)),
      A1host(new Vector(0)),
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
      Put(new Vector(0)),
      residM(0.0),
      residD(_alphaF),
      residR(_alphaF),
      residP(_alphaF),
      operatorsBuilt(false),
      motionNeedsGather(true),
      deviceMotionStale(true),
      velRecoverPending(false),
      m_impl(nullptr)
{
}

CudaExplicitAlpha_TP::~CudaExplicitAlpha_TP()
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
    delete Put;
    delete Vrep;
    delete Vforce;
    delete dVres;
    delete A1host;
}

bool CudaExplicitAlpha_TP::areAlphaMFClose() const
{
    return useAlphaCloseCheck && std::fabs(alphaM - alphaF) <= toleranceAlphaMF;
}

int CudaExplicitAlpha_TP::validateCudaSOE(CudaBcsrLinSOE *&cudaSOE) const
{
    LinearSOE *soe = this->getLinearSOE();
    if (soe == nullptr) {
        opserr << "ERROR CudaExplicitAlpha_TP - no LinearSOE linked; use `system CuDSS` or `system DistributedCuDSS`.\n";
        return -1;
    }
    cudaSOE = dynamic_cast<CudaBcsrLinSOE *>(soe);
    if (cudaSOE == nullptr) {
        if (auto *dist = dynamic_cast<DistributedCudaBcsrLinSOE *>(soe))
            cudaSOE = dist->getCudaBcsrLinSOE();
    }
    if (cudaSOE == nullptr || cudaSOE->getBlockSize() != 1 || cudaSOE->getCudaBcsrLinSolver() == nullptr) {
        opserr << "ERROR CudaExplicitAlpha_TP (CudaKRAlpha_TP/CudaMKRAlpha_TP) requires `system CuDSS` or `system DistributedCuDSS`.\n";
        return -2;
    }
    if (!isUniformPrecision(cudaSOE->getPrecision())) {
        opserr << "ERROR CudaExplicitAlpha_TP - unsupported CuDSS precision "
               << cudaPrecisionToString(cudaSOE->getPrecision())
               << "; only uniform dDDI and dFFI are supported.\n";
        return -3;
    }
    return 0;
}

int CudaExplicitAlpha_TP::formTangentIntoSOE(int statFlag, double c1v, double c2v, double c3v)
{
    if (this->getAnalysisModel() == nullptr) {
        return -1;
    }
    LinearSOE *soe = this->getLinearSOE();
    if (soe == nullptr) {
        return -1;
    }
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
        if (CudaBcsrLinSOE *cuda = dist->getCudaBcsrLinSOE()) {
            cuda->setAValuesPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
        }
    }
    return 0;
}

int CudaExplicitAlpha_TP::formTangent(int statFlag)
{
    (void)statFlag;
    return 0;  // operators assembled in formOperators(); Linear::formTangent is a no-op
}

int CudaExplicitAlpha_TP::formEleTangent(FE_Element *theEle)
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

int CudaExplicitAlpha_TP::formNodTangent(DOF_Group *theDof)
{
    if (operatorsBuilt) {
        return 0;
    }
    theDof->zeroTangent();
    theDof->addCtoTang(c2);
    theDof->addMtoTang(c3);
    return 0;
}

// TP residual weights (residM/D/R/P) switch between the two passes in newStep().
int CudaExplicitAlpha_TP::formEleResidual(FE_Element *theEle)
{
    theEle->zeroResidual();
    theEle->addRIncInertiaToResidual(residR);
    theEle->addM_Force(*Udotdot, residR - residM);
    return 0;
}

int CudaExplicitAlpha_TP::formNodUnbalance(DOF_Group *theDof)
{
    theDof->zeroUnbalance();
    theDof->addPtoUnbalance(residP);
    theDof->addD_Force(order2 ? *Vforce : *Udot, -residD);
    theDof->addM_Force(*Udotdot, -residM);
    return 0;
}

int CudaExplicitAlpha_TP::domainChanged()
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
        cudaSOE->setXSyncMode(false);
        m_impl->destroySolvers();
        m_impl->allocate(size, (areAlphaMFClose() || incrementalAccel) ? 1 : 2);
        const ImplBase::HostMotionPtrs buf = m_impl->ensureHostMotionBuffers(size);
        Ut->setData(buf.Ut, size);
        Utdot->setData(buf.Utdot, size);
        Utdotdot->setData(buf.Utdotdot, size);
        U->setData(buf.U, size);
        Udot->setData(buf.Udot, size);
        Udotdot->setData(buf.Uddot, size);
        Put->setData(m_impl->getPutHostPtr(), size);
    } else {
        destroyDeviceImpl();
        Ut->resize(size); Utdot->resize(size); Utdotdot->resize(size);
        U->resize(size); Udot->resize(size); Udotdot->resize(size);
        Put->resize(size);
        Ut->Zero(); Utdot->Zero(); Utdotdot->Zero();
        U->Zero(); Udot->Zero(); Udotdot->Zero(); Put->Zero();
    }
    operatorsBuilt = false;

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
    // -order2 mid-analysis: the domain holds the reported velocity v_alg + gamma*dt*alpha1*a; the
    // algorithmic one is recovered in order2PreStep, which forms dv = gamma*dt*alpha1*a_n.
    velRecoverPending = (order2 && !startPending && deltaT > 0.0);
    Vrep->resize(size);
    Vforce->resize(size);
    dVres->resize(size);
    A1host->resize(size);
    Vrep->Zero();
    Vforce->Zero();
    dVres->Zero();
    A1host->Zero();
    *Ut = *U;
    *Utdot = *Udot;
    *Utdotdot = *Udotdot;
    motionNeedsGather = true;
    deviceMotionStale = true;
    return 0;
}

// -initAccel / -order2 starting correction (collective in OpenSeesMP), TP residual weights of the n end.
int CudaExplicitAlpha_TP::applyStartCorrection(CudaBcsrLinSOE *cudaSOE)
{
    AnalysisModel *theModel = this->getAnalysisModel();
    auto *distSOE = dynamic_cast<DistributedCudaBcsrLinSOE *>(this->getLinearSOE());
    residD = residR = residP = 1.0;
    residM = 0.0;
    *Vrep = *Udot;    // prescribed v0
    *Vforce = *Udot;  // nodal damping at v0
    Udotdot->Zero();  // zero trial accel: residual p - C v0 - f
    deviceMotionStale = true;
    // Elements that update from the trial-displacement increment (forceBeamColumn) skip their update
    // when the increment is zero, so after setNodeDisp -commit they still hold the previous state and
    // R0 would miss the imposed displacement. A tiny non-committing trial excursion makes them iterate
    // to the committed configuration before R0 is formed.
    {
        Vector Up(*U);
        for (int i = 0; i < Up.Size(); i++)
            Up(i) += 1.0e-9 * (1.0 + fabs(Up(i)));
        theModel->setResponse(Up, *Udot, *Udotdot);
        if (theModel->updateDomain(theModel->getCurrentDomainTime(), deltaT) < 0)
            return -2;
    }
    theModel->setResponse(*U, *Udot, *Udotdot);
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
        if (distSOE->getProcessID() == 0) {
            if (rc == 0)
                rc = (m_impl != nullptr) ? m_impl->applyStart(this, cudaSOE, &(*A1host)(0)) : -5;
            status(0) = (rc == 0) ? 1.0 : -1.0;
        }
        if (distSOE->broadcastFromRoot(status) < 0)
            return -3;
        if (status(0) < 0.0)
            return -4;
        if (distSOE->broadcastFromRoot(*Udot) < 0 || distSOE->broadcastFromRoot(*Udotdot) < 0 ||
            distSOE->broadcastFromRoot(*A1host) < 0)
            return -3;
    } else {
        if (m_impl == nullptr)
            return -5;
        rc = m_impl->applyStart(this, cudaSOE, &(*A1host)(0));
        if (rc != 0)
            return rc;
    }
    if (order2) {
        // n end of the first step: v_alg0 + dv_0 (= v0) and a0, so the inertia terms cancel
        *Vrep = *Udot;
        Vrep->addVector(1.0, *A1host, gamma * deltaT);
        theModel->setResponse(*U, *Vrep, *Udotdot);
    } else
        theModel->setResponse(*U, *Udot, *Udotdot);
    return 0;
}

// Per-step flow (host residual assembly + GPU predictor/RHS):
//   1) newStep: TransientIntegrator::formUnbalance at t_n   -> d_put <- device B^(1)
//   2) GPU predictor                                      -> trial (U, Udot, Uddot) at t_{n+1}
//   3) newStep: TransientIntegrator::formUnbalance at t_{n+1} -> blend Put with B^(2)
// Linear::solveCurrentStep then calls CudaExplicitAlpha_TP::formUnbalance() (Put -> B) and update() (accel only).
int CudaExplicitAlpha_TP::newStep(double _deltaT)
{
    updateCount = 0;
    if (alphaF < 0.5 || alphaF > 1.0 || beta <= 0.0 || gamma <= 0.0 || _deltaT <= 0.0) {
        return -1;
    }
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr) {
        opserr << "WARNING CudaExplicitAlpha_TP::newStep() - no AnalysisModel set\n";
        return -2;
    }
    CudaBcsrLinSOE *cudaSOE = nullptr;
    if (validateCudaSOE(cudaSOE) != 0) {
        return -3;
    }
    auto *distSOE = dynamic_cast<DistributedCudaBcsrLinSOE *>(this->getLinearSOE());
    // Drop any leftover B-skip flag from a prior formUnbalance that never reached solve().
    if (distSOE != nullptr)
        distSOE->setGlobalRhsOnDevice(false);

    const bool deviceOn = cudaSOE->isCudaDeviceEnabled();
    if (deviceOn) {
        ensureDeviceImpl(cudaSOE);
        if (m_impl == nullptr) {
            opserr << "ERROR CudaExplicitAlpha_TP::newStep() - GPU state not initialized; "
                   << "rebuild analysis with `system CuDSS`.\n";
            return -2;
        }
        cudaSOE->setXSyncMode(false);
    } else if (distSOE == nullptr) {
        opserr << "ERROR CudaExplicitAlpha_TP::newStep() - GPU state not initialized; "
               << "rebuild analysis with `system CuDSS`.\n";
        return -2;
    }

    // Same gather as CudaExplicitAlpha: P0 needs full motion before GPU work / broadcast.
    if (distSOE != nullptr && motionNeedsGather) {
        if (distSOE->gatherOwnedToRoot(*U) < 0 ||
            distSOE->gatherOwnedToRoot(*Udot) < 0 ||
            distSOE->gatherOwnedToRoot(*Udotdot) < 0) {
            opserr << "ERROR CudaExplicitAlpha_TP::newStep() - motion gather to root failed\n";
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
        if (formOperators(cudaSOE) != 0) {
            return -4;
        }
        operatorsBuilt = true;
    }

    if (initAccel && startPending) {
        if (applyStartCorrection(cudaSOE) != 0) {
            opserr << "ERROR CudaExplicitAlpha_TP::newStep() - starting correction (-initAccel / -order2) failed\n";
            return -10;
        }
        startPending = false;
        *Ut = *U;
        *Utdot = *Udot;
        *Utdotdot = *Udotdot;
    }
    if (order2) {
        // dv before the n-end pass (root GPU; broadcast to workers)
        int rc = 0;
        if (deviceOn)
            rc = (m_impl != nullptr) ? m_impl->order2PreStep(this, &(*dVres)(0)) : -1;
        if (distSOE != nullptr && distSOE->broadcastFromRoot(*dVres) < 0)
            return -11;
        if (rc != 0)
            return -11;
        // n end: committed state (velocity = reported v_alg_n + dv); nodal damping likewise
        *Vforce = *Utdot;
        Vforce->addVector(1.0, *dVres, 1.0);
    }

    // --- First pass at t_n: no explicit nodal M*a term (residM = 0) ---
    residD = residR = residP = 1.0;
    residM = 0.0;

    double time = theModel->getCurrentDomainTime();
    if (theModel->updateDomain(time, deltaT) < 0) {
        return -5;
    }
    if (TransientIntegrator::formUnbalance() < 0) {
        return -6;
    }
    if (distSOE != nullptr) {
        {
            if (distSOE->mergeBToRoot() < 0)
                return -6;
        }
        distSOE->zeroB();
        if (distSOE->getProcessID() != 0)
            ; // workers: merged B lives on P0 only
        else {
            cudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
            if (m_impl->capturePutFromDeviceB(cudaSOE) != 0)
                return -6;
        }
    } else if (m_impl->capturePutFromDeviceB(cudaSOE) != 0) {
        return -6;
    }

    if (deviceOn) {
        if (m_impl->newStepPredictor(this) != 0) {
            return -7;
        }
    }
    if (distSOE != nullptr) {
        if (distSOE->broadcastFromRoot(*U) < 0 ||
            distSOE->broadcastFromRoot(*Udot) < 0 ||
            distSOE->broadcastFromRoot(*Udotdot) < 0) {
            return -7;
        }
    } else if (!deviceOn) {
        return -7;
    }

    // Full-step trial kinematics for the second pass (not alpha-interpolated).
    if (order2) {
        // n+1 end: v_alg_{n+1} + dv (a_n stands in for a_{n+1})
        *Vforce = *Udot;
        Vforce->addVector(1.0, *dVres, 1.0);
        theModel->setResponse(*U, *Vforce, *Udotdot);
    } else
        theModel->setResponse(*U, *Udot, *Udotdot);

    // --- Second pass at t_{n+1}: include nodal M*a (residM = 1) ---
    time += deltaT;
    if (theModel->updateDomain(time, deltaT) < 0) {
        return -8;
    }

    residM = 1.0;
    if (TransientIntegrator::formUnbalance() < 0) {
        return -9;
    }

    if (distSOE != nullptr) {
        {
            if (distSOE->mergeBToRoot() < 0)
                return -9;
        }
        distSOE->zeroB();
        if (distSOE->getProcessID() != 0)
            ;
        else {
            cudaSOE->setBPrimaryLocation(CudaBcsrLinSOE::DataLocation::Host);
            m_impl->blendPutFromHostB(cudaSOE, 1.0 - alphaF, alphaF);
            if (order2 && m_impl->addRateShiftPut() != 0)
                return -12;
        }
    } else {
        m_impl->blendPutFromHostB(cudaSOE, 1.0 - alphaF, alphaF);
        if (order2 && m_impl->addRateShiftPut() != 0)
            return -12;  // -order2: Put += C_int*dv
    }
    return 0;
}

int CudaExplicitAlpha_TP::formUnbalance()
{
    CudaBcsrLinSOE *cudaSOE = nullptr;
    if (validateCudaSOE(cudaSOE) != 0) {
        return -2;
    }
    auto *distSOE = dynamic_cast<DistributedCudaBcsrLinSOE *>(this->getLinearSOE());
    if (distSOE != nullptr) {
        distSOE->setGlobalRhsOnDevice(false);
        distSOE->zeroB();

        // Status broadcast keeps every rank's globalRhsOnDevice flag matched so
        // DistCuDSS::solve collectives cannot diverge on a root-only GPU failure.
        Vector status(1);
        if (distSOE->getProcessID() != 0) {
            if (distSOE->broadcastFromRoot(status) < 0)
                return -2;
            if (status(0) < 0.0)
                return -2;
            distSOE->setGlobalRhsOnDevice(true);
            return 0;
        }

        int rc = -1;
        if (m_impl != nullptr && cudaSOE->isCudaDeviceEnabled())
            rc = m_impl->formUnbalanceFromPut(cudaSOE, areAlphaMFClose());

        status(0) = (rc == 0) ? 1.0 : -1.0;
        if (distSOE->broadcastFromRoot(status) < 0)
            return -2;
        if (rc == 0)
            distSOE->setGlobalRhsOnDevice(true);
        return rc;
    }
    if (m_impl == nullptr)
        return -1;
    return m_impl->formUnbalanceFromPut(cudaSOE, areAlphaMFClose());
}

int CudaExplicitAlpha_TP::update(const Vector &aiPlusOne)
{
    updateCount++;
    if (updateCount > 1) {
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
        if (m_impl == nullptr || m_impl->updateState(cudaSOE, incrementalAccel, alphaF) != 0) {
            return -4;
        }
    } else {
        // Worker: X already broadcast on host by DistributedCudaBcsrLinSOE::solve.
        if (incrementalAccel) {
            Udotdot->addVector(alphaF, aiPlusOne, 1.0);
        } else {
            *Udotdot = aiPlusOne;
        }
    }
    if (order2) {
        // Commit the reported velocity v_alg + gamma*dt*alpha1*a: also the next n end.
        *Vrep = *Udot;
        int rc = 0;
        if (cudaSOE->isCudaDeviceEnabled())
            rc = (m_impl != nullptr) ? m_impl->alpha1Accel(&(*A1host)(0)) : -1;
        if (auto *distSOE = dynamic_cast<DistributedCudaBcsrLinSOE *>(this->getLinearSOE())) {
            if (distSOE->broadcastFromRoot(*A1host) < 0)
                return -5;
        }
        if (rc != 0)
            return -5;
        Vrep->addVector(1.0, *A1host, gamma * deltaT);
        theModel->setVel(*Vrep);
    }
    theModel->setAccel(*Udotdot);
    if (theModel->updateDomain() < 0) {
        return -3;
    }
    return 0;
}

int CudaExplicitAlpha_TP::commit()
{
    AnalysisModel *theModel = this->getAnalysisModel();
    if (theModel == nullptr) {
        return -1;
    }
    // Domain time was advanced to t_{n+1} during the second updateDomain in newStep().
    return theModel->commitDomain();
}

int CudaExplicitAlpha_TP::revertToLastStep()
{
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

double CudaExplicitAlpha_TP::getCFactor() { return c2; }

const Vector &CudaExplicitAlpha_TP::getVel() { return order2 ? *Vforce : *Udot; }  // end-of-step velocity, not alpha-level

int CudaExplicitAlpha_TP::sendSelf(int cTag, Channel &theChannel)
{
    Vector data(8);
    data(0) = alphaM;
    data(1) = alphaF;
    data(2) = beta;
    data(3) = gamma;
    data(4) = incrementalAccel ? 1.0 : 0.0;
    data(5) = useAlphaCloseCheck ? 1.0 : 0.0;
    data(6) = order2 ? 1.0 : 0.0;
    data(7) = initAccel ? 1.0 : 0.0;
    return theChannel.sendVector(this->getDbTag(), cTag, data);
}

int CudaExplicitAlpha_TP::recvSelf(int cTag, Channel &theChannel, FEM_ObjectBroker &theBroker)
{
    Vector data(8);
    if (theChannel.recvVector(this->getDbTag(), cTag, data) < 0) {
        return -1;
    }
    order2 = data(6) > 0.5;
    initAccel = data(7) > 0.5 || order2;
    alphaM = data(0);
    alphaF = data(1);
    beta = data(2);
    gamma = data(3);
    incrementalAccel = data(4) > 0.5;
    useAlphaCloseCheck = data(5) > 0.5;
    residM = 0.0;
    residD = alphaF;
    residR = alphaF;
    residP = alphaF;
    operatorsBuilt = false;
    return 0;
}

void CudaExplicitAlpha_TP::Print(OPS_Stream &s, int flag)
{
    (void)flag;
    s << "CudaExplicitAlpha_TP alphaF=" << alphaF << " alphaM=" << alphaM
      << " order2=" << (order2 ? 1 : 0) << " initAccel=" << (initAccel ? 1 : 0);
    s << endln;
}

int CudaExplicitAlpha_TP::revertToStart()
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
        Put->Zero();
        if (m_impl != nullptr && m_impl->getSize() > 0) {
            m_impl->zeroState();
        }
    }
    startPending = true;
    deviceMotionStale = true;
    return 0;
}
