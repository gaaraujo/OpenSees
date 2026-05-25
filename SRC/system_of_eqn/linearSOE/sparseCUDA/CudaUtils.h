/* ****************************************************************** **
**    OpenSees - Open System for Earthquake Engineering Simulation    **
**          Pacific Earthquake Engineering Research Center            **
**                                                                    **
** Common CUDA utility functions for error checking and helpers       **
**                                                                    **
** ****************************************************************** */

#ifndef CudaUtils_h
#define CudaUtils_h

#include <OPS_Globals.h>
#include <stdexcept>
#include <string>
#include <cstring>

// ============================================================================
// Precision Type Definitions
// ============================================================================

// Precision modes for CUDA solvers and SOEs (matches AMGX naming convention)
// Format: d<Matrix><Vector>I where:
//   - d: device (GPU)
//   - Matrix: D (double) or F (float) for matrix values
//   - Vector: D (double) or F (float) for vector values
//   - I: Int (32-bit integer indices)
enum class CudaPrecision {
    dDDI,  // device, Double matrix, Double vector, Int index
    dFFI,  // device, Float matrix, Float vector, Int index
    dDFI,  // device, Double matrix, Float vector, Int index
    dFDI   // device, Float matrix, Double vector, Int index
};

// Conversion utilities for CudaPrecision
inline const char* cudaPrecisionToString(CudaPrecision precision) {
    switch(precision) {
        case CudaPrecision::dDDI: return "dDDI";
        case CudaPrecision::dFFI: return "dFFI";
        case CudaPrecision::dDFI: return "dDFI";
        case CudaPrecision::dFDI: return "dFDI";
        default: return "unknown";
    }
}

inline bool cudaPrecisionFromString(const char* str, CudaPrecision& precision) {
    if (std::strcmp(str, "dDDI") == 0) {
        precision = CudaPrecision::dDDI;
        return true;
    } else if (std::strcmp(str, "dFFI") == 0) {
        precision = CudaPrecision::dFFI;
        return true;
    } else if (std::strcmp(str, "dDFI") == 0) {
        precision = CudaPrecision::dDFI;
        return true;
    } else if (std::strcmp(str, "dFDI") == 0) {
        precision = CudaPrecision::dFDI;
        return true;
    }
    return false;
}

inline bool isDoublePrecision(CudaPrecision precision) {
    return precision == CudaPrecision::dDDI;
}

// Check if precision uses uniform types (matrix == vector)
// Note: !isUniformPrecision(p) means mixed precision
inline bool isUniformPrecision(CudaPrecision precision) {
    return precision == CudaPrecision::dDDI || precision == CudaPrecision::dFFI;
}

// ============================================================================
// CUDA Error Checking
// ============================================================================

#ifdef _CUDA
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

namespace CudaUtils {

// CUDA error checking
inline void cudaCheckError(cudaError_t error, const char* message, bool throwError = true)
{
    if (error != cudaSuccess) {
        const char* errorString = cudaGetErrorString(error);
        if (throwError) {
            throw std::runtime_error(
                "CUDA API returned error " + 
                std::string(errorString) + 
                " for " + std::string(message)
            );
        } else {
            opserr << "CUDA API returned error " << errorString << " for " << message << endln;
        }
    }
}

// cuSPARSE error checking
inline void cuSparseCheckError(cusparseStatus_t error, const char* message, bool throwError = true)
{
    if (error != CUSPARSE_STATUS_SUCCESS) {
        if (throwError) {
            throw std::runtime_error(
                "cuSPARSE API returned error " + 
                std::to_string(error) + 
                " for " + std::string(message)
            );
        } else {
            opserr << "cuSPARSE API returned error " << error << " for " << message << endln;
        }
    }
}

// cuBLAS error checking
inline void cublasCheckError(cublasStatus_t error, const char* message, bool throwError = true)
{
    if (error != CUBLAS_STATUS_SUCCESS) {
        if (throwError) {
            throw std::runtime_error(
                "cuBLAS API returned error " + 
                std::to_string(error) + 
                " for " + std::string(message)
            );
        } else {
            opserr << "cuBLAS API returned error " << error << " for " << message << endln;
        }
    }
}

// Type-safe cuBLAS wrappers - automatically dispatch based on type
inline cublasStatus_t cublasDot(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result) {
    return cublasDdot(handle, n, x, incx, y, incy, result);
}

inline cublasStatus_t cublasDot(cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result) {
    return cublasSdot(handle, n, x, incx, y, incy, result);
}

inline cublasStatus_t cublasAxpy(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy) {
    return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

inline cublasStatus_t cublasAxpy(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy) {
    return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

inline cublasStatus_t cublasScal(cublasHandle_t handle, int n, const double* alpha, double* x, int incx) {
    return cublasDscal(handle, n, alpha, x, incx);
}

inline cublasStatus_t cublasScal(cublasHandle_t handle, int n, const float* alpha, float* x, int incx) {
    return cublasSscal(handle, n, alpha, x, incx);
}

inline cublasStatus_t cublasNrm2(cublasHandle_t handle, int n, const double* x, int incx, double* result) {
    return cublasDnrm2(handle, n, x, incx, result);
}

inline cublasStatus_t cublasNrm2(cublasHandle_t handle, int n, const float* x, int incx, float* result) {
    return cublasSnrm2(handle, n, x, incx, result);
}

} // namespace CudaUtils

#endif // _CUDA

#ifdef _CUDSS
#include <cudss.h>

namespace CudaUtils {

// cuDSS error checking
inline void cuDSSCheckError(cudssStatus_t error, const char* message, bool throwError = true)
{
    if (error != CUDSS_STATUS_SUCCESS) {
        if (throwError) {
            throw std::runtime_error(
                "cuDSS API returned error " + 
                std::to_string(error) + 
                " for " + std::string(message)
            );
        } else {
            opserr << "cuDSS API returned error " << error << " for " << message << endln;
        }
    }
}

} // namespace CudaUtils

#endif // _CUDSS

#endif // CudaUtils_h

