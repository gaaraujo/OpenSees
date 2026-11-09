#include "SparsePythonEigenFactory.h"

#include "DistributedSparsePythonEigenSOE.h"
#include "SparsePythonCompressedEigenSOE.h"
#include "SparsePythonCompressedEigenSolver.h"
#include "SparsePythonCOOEigenSOE.h"
#include "SparsePythonCOOEigenSolver.h"
#include "SparsePythonEigenCommon.h"

#include <OPS_Globals.h>
#include <elementAPI.h>

#include <Python.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>

namespace {

bool g_sparsePythonEigenDistributed = false;

bool
ParseSchemeToken(const char *token, SparsePythonEigenStorageScheme &scheme)
{
    if (token == nullptr) {
        return false;
    }
    std::string normalized(token);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (normalized == "csr" || normalized == "row") {
        scheme = SparsePythonEigenStorageScheme::CSR;
        return true;
    }
    if (normalized == "csc" || normalized == "column" || normalized == "col") {
        scheme = SparsePythonEigenStorageScheme::CSC;
        return true;
    }
    if (normalized == "coo") {
        scheme = SparsePythonEigenStorageScheme::COO;
        return true;
    }
    return false;
}

} // namespace

void
OPS_SetSparsePythonEigenDistributed(bool distributed)
{
    g_sparsePythonEigenDistributed = distributed;
}

void *
OPS_SparsePythonEigenSolver()
{
    const char *expectedSyntax =
        "eigen 'PythonSparse'|'DistributedPythonSparse' numModes "
        "{'solver': SolverObject, 'scheme': 'CSR'|'CSC'|'COO'}";

    // Note: This function is called AFTER numModes has been read by the caller.
    // Call OPS_SetSparsePythonEigenDistributed(true) before invoking for
    // DistributedPythonSparse so workers may omit the Python solver object.

    if (!Py_IsInitialized()) {
        Py_Initialize();
    }

    PyGILState_STATE gilState = PyGILState_Ensure();

    const bool distributed = g_sparsePythonEigenDistributed;
    // Reset so a subsequent serial eigen call does not inherit the flag.
    g_sparsePythonEigenDistributed = false;

    const char *typeName = distributed ? "DistributedPythonSparse" : "PythonSparse";

    void *dictPtr = OPS_GetVoidPtr();
    if (dictPtr == nullptr) {
        opserr << "WARNING: eigen " << typeName << " - requires a dictionary argument as third parameter" << endln;
        opserr << "Expected syntax: " << expectedSyntax << endln;
        PyGILState_Release(gilState);
        return nullptr;
    }

    PyObject *dict = static_cast<PyObject *>(dictPtr);
    if (!PyDict_Check(dict)) {
        opserr << "WARNING: eigen " << typeName << " - third argument must be a dictionary" << endln;
        opserr << "Expected syntax: " << expectedSyntax << endln;
        PyGILState_Release(gilState);
        return nullptr;
    }

    PyObject *solverObj = PyDict_GetItemString(dict, "solver");
    if (solverObj == Py_None) {
        solverObj = nullptr;
    }
    if (!distributed && solverObj == nullptr) {
        opserr << "WARNING: eigen " << typeName << " - dictionary must contain 'solver' key" << endln;
        opserr << "Expected syntax: " << expectedSyntax << endln;
        PyGILState_Release(gilState);
        return nullptr;
    }

    SparsePythonEigenStorageScheme scheme = SparsePythonEigenStorageScheme::CSR;
    PyObject *schemeObj = PyDict_GetItemString(dict, "scheme");
    if (schemeObj != nullptr) {
        if (!PyUnicode_Check(schemeObj)) {
            opserr << "WARNING: eigen " << typeName << " - 'scheme' must be a string" << endln;
            PyGILState_Release(gilState);
            return nullptr;
        }
        const char *schemeStr = PyUnicode_AsUTF8(schemeObj);
        if (schemeStr == nullptr || !ParseSchemeToken(schemeStr, scheme)) {
            opserr << "WARNING: eigen " << typeName << " - unknown storage scheme '"
                   << (schemeStr != nullptr ? schemeStr : "null") << "' (expected CSR, CSC, or COO)" << endln;
            PyGILState_Release(gilState);
            return nullptr;
        }
    }

    void *result = nullptr;
    if (scheme == SparsePythonEigenStorageScheme::COO) {
        if (solverObj == nullptr) {
            result = new SparsePythonCOOEigenSOE();
        } else {
            SparsePythonCOOEigenSolver *solver = new SparsePythonCOOEigenSolver();
            if (solver->setPythonCallable(solverObj, "solve") != 0) {
                opserr << "WARNING: eigen " << typeName << " - failed to set Python callable" << endln;
                delete solver;
                PyGILState_Release(gilState);
                return nullptr;
            }
            result = new SparsePythonCOOEigenSOE(*solver);
        }
    } else {
        if (solverObj == nullptr) {
            result = new SparsePythonCompressedEigenSOE(scheme);
        } else {
            SparsePythonCompressedEigenSolver *solver = new SparsePythonCompressedEigenSolver();
            if (solver->setPythonCallable(solverObj, "solve") != 0) {
                opserr << "WARNING: eigen " << typeName << " - failed to set Python callable" << endln;
                delete solver;
                PyGILState_Release(gilState);
                return nullptr;
            }
            result = new SparsePythonCompressedEigenSOE(*solver, scheme);
        }
    }

    if (result != nullptr && distributed) {
        result = new DistributedSparsePythonEigenSOE(static_cast<EigenSOE *>(result));
    }

    PyGILState_Release(gilState);
    return result;
}
