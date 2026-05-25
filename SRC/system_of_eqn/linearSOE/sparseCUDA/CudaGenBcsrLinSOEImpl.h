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
                                                                        
// $Source: OpenSees/SRC/system_of_eqn/linearSOE/sparseCuda/CudaGenBcsrLinSOEImpl.h
                                                                        
// Written: gaaraujo 
// Created: 08/2025
//
// Description: This file contains the internal template implementation for 
// CudaGenBcsrLinSOE. This is an internal implementation detail and should
// not be included by users of the CudaGenBcsrLinSOE interface.
//

#ifndef CudaGenBcsrLinSOEImpl_h
#define CudaGenBcsrLinSOEImpl_h

// OpenSees includes
#include <CudaGenBcsrLinSOE.h>
#include <CudaGenBcsrLinSolver.h>
#ifdef _CUDA
// CUDA includes
#include <cuda_runtime.h>

// Thrust includes
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/memory.h>
#include <thrust/system/cuda/execution_policy.h>

// Bring thrust::raw_pointer_cast into scope for CUDA builds
using thrust::raw_pointer_cast;
#else
#include <vector>
#include <algorithm>

// Define a passthrough raw_pointer_cast for non-CUDA builds
template<typename T>
inline T* raw_pointer_cast(T* ptr) { return ptr; }
#endif

// C++ includes
#include <type_traits>

// Forward declarations
class CudaGenBcsrLinSolver;

#ifdef _CUDA
namespace {
template<typename VectorType>
struct AddDoubleToB {
    __device__ __host__ VectorType operator()(double a, VectorType b) const { return b + static_cast<VectorType>(a); }
};
}
#endif

// This template class provides the actual implementation for different data types.
// It inherits from CudaGenBcsrLinSOE and implements all the pure virtual methods.
// The template parameters allow us to specialize matrix and vector types independently:
//   - MatrixType: Type for matrix values (A)
//   - VectorType: Type for vectors (x, b)
// 
// Currently instantiated combinations:
//   - CudaGenBcsrLinSOEImpl<double, double> (dDDI) - uniform double precision
//   - CudaGenBcsrLinSOEImpl<float, float>   (dFFI) - uniform float precision
// 
// Additional combinations can be instantiated when needed:
//   - CudaGenBcsrLinSOEImpl<double, float>  (dDFI) - double matrix, float vectors
//   - CudaGenBcsrLinSOEImpl<float, double>  (dFDI) - float matrix, double vectors
// 
// Note: Whether a precision mode is "supported" depends on the solver, not the SOE.
//       The SOE simply provides the data in the requested format.
template<typename MatrixType, typename VectorType = MatrixType>
class CudaGenBcsrLinSOEImpl : public CudaGenBcsrLinSOE
{
public:
    // Constructor with solver
    explicit CudaGenBcsrLinSOEImpl(
        CudaGenBcsrLinSolver &theSolver, 
        int blockSize = DEFAULT_BLOCK_SIZE, 
        bool paddingEnabled = true,
        bool verbose = false,
        bool symmetricStorage = false
    )
    : CudaGenBcsrLinSOE(CudaGenBcsrLinSOEImpl<MatrixType, VectorType>::getClassTagForType(), theSolver, blockSize, paddingEnabled, verbose, symmetricStorage),
      m_deviceAValues(), m_deviceX(), m_deviceB()
    {
        // Now that the derived class is fully constructed, we can safely call setLinearSOE
        theSolver.setLinearSOE(*this);
    }
    
    // Default constructor
    CudaGenBcsrLinSOEImpl()
    : CudaGenBcsrLinSOE(CudaGenBcsrLinSOEImpl<MatrixType, VectorType>::getClassTagForType()),
      m_deviceAValues(), m_deviceX(), m_deviceB()
    {
        /* Nothing to do here */
    }

    // Destructor
    ~CudaGenBcsrLinSOEImpl() = default;

    // Disable copy constructor and assignment
    CudaGenBcsrLinSOEImpl(const CudaGenBcsrLinSOEImpl&) = delete;
    CudaGenBcsrLinSOEImpl& operator=(const CudaGenBcsrLinSOEImpl&) = delete;

    // Move constructor and assignment
    CudaGenBcsrLinSOEImpl(CudaGenBcsrLinSOEImpl&&) = default;
    CudaGenBcsrLinSOEImpl& operator=(CudaGenBcsrLinSOEImpl&&) = default;

    // Required methods for CudaGenBcsrLinSOE subclasses
    // These methods provide type-erased access to device data.
    // The void* return type allows the solver to work with any data type without knowing the specifics.
    const void* getDeviceAValues(void) const noexcept override { 
        #ifdef _CUDA
        return m_deviceAValues.empty() ? nullptr : m_deviceAValues.data().get(); 
        #else
        return m_deviceAValues.empty() ? nullptr : m_deviceAValues.data(); 
        #endif
    }
    
    void* getDeviceAValues(void) noexcept override { 
        #ifdef _CUDA
        return m_deviceAValues.empty() ? nullptr : m_deviceAValues.data().get(); 
        #else
        return m_deviceAValues.empty() ? nullptr : m_deviceAValues.data(); 
        #endif
    }
    
    const void* getDeviceX(void) const noexcept override { 
        #ifdef _CUDA
        return m_deviceX.empty() ? nullptr : m_deviceX.data().get(); 
        #else
        return m_deviceX.empty() ? nullptr : m_deviceX.data(); 
        #endif
    }
    
    void* getDeviceX(void) noexcept override { 
        #ifdef _CUDA
        return m_deviceX.empty() ? nullptr : m_deviceX.data().get(); 
        #else
        return m_deviceX.empty() ? nullptr : m_deviceX.data(); 
        #endif
    }
    
    const void* getDeviceB(void) const noexcept override { 
        #ifdef _CUDA
        return m_deviceB.empty() ? nullptr : m_deviceB.data().get(); 
        #else
        return m_deviceB.empty() ? nullptr : m_deviceB.data(); 
        #endif
    }
    
    void* getDeviceB(void) noexcept override { 
        #ifdef _CUDA
        return m_deviceB.empty() ? nullptr : m_deviceB.data().get(); 
        #else
        return m_deviceB.empty() ? nullptr : m_deviceB.data(); 
        #endif
    }
    
    // Precision query method
    CudaPrecision getPrecision(void) const noexcept override {
        // Determine precision from template types
        constexpr bool matrixDouble = std::is_same<MatrixType, double>::value;
        constexpr bool vectorDouble = std::is_same<VectorType, double>::value;
        
        if constexpr (matrixDouble && vectorDouble) {
            return CudaPrecision::dDDI;  // Double matrix, Double vectors
        } else if constexpr (!matrixDouble && !vectorDouble) {
            return CudaPrecision::dFFI;  // Float matrix, Float vectors
        } else if constexpr (matrixDouble && !vectorDouble) {
            return CudaPrecision::dDFI;  // Double matrix, Float vectors
        } else {  // !matrixDouble && vectorDouble
            return CudaPrecision::dFDI;  // Float matrix, Double vectors
        }
    }
    
    // Host (double)-device (VectorType) data transfer methods
    inline void uploadVectorsToDevice(void) override {
        #ifdef _CUDA
        // Use thrust for transfer (handles type conversion)
        m_deviceB = this->CudaGenBcsrLinSOE::m_hostB;
        m_deviceX.resize(this->CudaGenBcsrLinSOE::m_hostX.size());
        #else
        m_deviceB.resize(this->CudaGenBcsrLinSOE::m_hostB.size());
        std::transform(
            this->CudaGenBcsrLinSOE::m_hostB.begin(),
            this->CudaGenBcsrLinSOE::m_hostB.end(),
            m_deviceB.begin(),
            [](double val){ return static_cast<VectorType>(val); } // convert to device type
        );
        m_deviceX.resize(this->CudaGenBcsrLinSOE::m_hostX.size());
        #endif
    }
    
    inline void downloadSolutionFromDevice(void) override {
        #ifdef _CUDA
        // Use thrust for transfer (handles type conversion)
        this->CudaGenBcsrLinSOE::m_hostX = m_deviceX;
        #else
        this->CudaGenBcsrLinSOE::m_hostX.resize(m_deviceX.size());
        std::transform(
            m_deviceX.begin(),
            m_deviceX.end(),
            this->CudaGenBcsrLinSOE::m_hostX.begin(),
            [](VectorType val){ return static_cast<double>(val); } // convert to host type
        );
        #endif
        this->CudaGenBcsrLinSOE::m_X.setData(
            raw_pointer_cast(this->CudaGenBcsrLinSOE::m_hostX.data()), 
            this->CudaGenBcsrLinSOE::m_X.Size()
        );
    }
    
    inline void uploadAValuesToDevice(void) override {
        #ifdef _CUDA
        // Use thrust for transfer (handles type conversion)
        m_deviceAValues = this->CudaGenBcsrLinSOE::m_hostAValues;
        #else
        m_deviceAValues.resize(this->CudaGenBcsrLinSOE::m_hostAValues.size());
        std::transform(
            this->CudaGenBcsrLinSOE::m_hostAValues.begin(),
            this->CudaGenBcsrLinSOE::m_hostAValues.end(),
            m_deviceAValues.begin(),
            [](double val){ return static_cast<MatrixType>(val); } // convert to device type
        );
        #endif
    }

    inline void downloadAValuesFromDevice(void) override {
        #ifdef _CUDA
        this->CudaGenBcsrLinSOE::m_hostAValues = m_deviceAValues;
        #else
        this->CudaGenBcsrLinSOE::m_hostAValues.resize(m_deviceAValues.size());
        std::transform(
            m_deviceAValues.begin(),
            m_deviceAValues.end(),
            this->CudaGenBcsrLinSOE::m_hostAValues.begin(),
            [](MatrixType val){ return static_cast<double>(val); }
        );
        #endif
    }

    inline void ensureDeviceVectorSizes(void) override {
        const size_t bSize = static_cast<size_t>(this->CudaGenBcsrLinSOE::m_hostB.size());
        const size_t xSize = static_cast<size_t>(this->CudaGenBcsrLinSOE::m_hostX.size());
        const size_t aSize = this->CudaGenBcsrLinSOE::m_hostAValues.size();
        m_deviceB.resize(bSize);
        m_deviceX.resize(xSize);
        m_deviceAValues.resize(aSize);
    }

    inline void addToDeviceBFromHost(int n, const double* hostData, void* stream) override {
        if (n <= 0 || hostData == nullptr) return;
        #ifdef _CUDA
        const size_t un = static_cast<size_t>(n);
        if (m_deviceB.size() < un) return;
        if (m_deviceStagingB.size() < un)
            m_deviceStagingB.resize(un);
        cudaStream_t s = (stream != nullptr) ? static_cast<cudaStream_t>(stream) : 0;
        cudaMemcpyAsync(raw_pointer_cast(m_deviceStagingB.data()), hostData, un * sizeof(double), cudaMemcpyHostToDevice, s);
        if (s != 0)
            thrust::transform(thrust::cuda::par.on(s),
                m_deviceStagingB.begin(), m_deviceStagingB.begin() + n,
                m_deviceB.begin(), m_deviceB.begin(), AddDoubleToB<VectorType>());
        else
            thrust::transform(thrust::cuda::par,
                m_deviceStagingB.begin(), m_deviceStagingB.begin() + n,
                m_deviceB.begin(), m_deviceB.begin(), AddDoubleToB<VectorType>());
        #else
        (void)stream;
        for (int i = 0; i < n && i < static_cast<int>(m_deviceB.size()); i++)
            m_deviceB[static_cast<size_t>(i)] += static_cast<VectorType>(hostData[i]);
        #endif
    }

private:    
    // Device memory storage using Thrust vectors
    #ifdef _CUDA
    thrust::device_vector<VectorType> m_deviceX, m_deviceB;
    thrust::device_vector<MatrixType> m_deviceAValues;
    thrust::device_vector<double> m_deviceStagingB;
    #else
    std::vector<VectorType> m_deviceX, m_deviceB;
    std::vector<MatrixType> m_deviceAValues;
    #endif

public:
    // Helper function to get class tag for type
    static int getClassTagForType() {
        constexpr bool matrixDouble = std::is_same_v<MatrixType, double>;
        constexpr bool vectorDouble = std::is_same_v<VectorType, double>;
        
        if constexpr (matrixDouble && vectorDouble) {
            return LinSOE_TAGS_CudaBcsrLinSOE_DOUBLE;        // dDDI
        } else if constexpr (!matrixDouble && !vectorDouble) {
            return LinSOE_TAGS_CudaBcsrLinSOE_FLOAT;         // dFFI
        } else if constexpr (matrixDouble && !vectorDouble) {
            return LinSOE_TAGS_CudaBcsrLinSOE_DOUBLE_FLOAT;  // dDFI
        } else {  // !matrixDouble && vectorDouble
            return LinSOE_TAGS_CudaBcsrLinSOE_FLOAT_DOUBLE;  // dFDI
        }
    }

private:
};

// Explicit template instantiations
// All four precision modes are instantiated and ready to use.
// Solvers validate which modes they support at construction/connection time.
template class CudaGenBcsrLinSOEImpl<double, double>;  // dDDI - uniform double precision
template class CudaGenBcsrLinSOEImpl<float, float>;    // dFFI - uniform float precision
template class CudaGenBcsrLinSOEImpl<double, float>;   // dDFI - double matrix, float vectors (mixed)
template class CudaGenBcsrLinSOEImpl<float, double>;   // dFDI - float matrix, double vectors (mixed)

#endif
