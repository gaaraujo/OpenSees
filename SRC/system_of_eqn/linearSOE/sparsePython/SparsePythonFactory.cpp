#include "SparsePythonFactory.h"

#include "SparsePythonCOOLinSOE.h"
#include "SparsePythonCOOLinSolver.h"
#include "SparsePythonCompressedLinSOE.h"
#include "SparsePythonCompressedLinSolver.h"
#include "SparsePythonCommon.h"

#include <OPS_Globals.h>
#include <elementAPI.h>

#include <Python.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>
#include <string>

namespace {

bool
IsSparsePythonType(const char *type)
{
    return strcmp(type, "PythonSparse") == 0;
}

std::string
ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool
ParseSchemeToken(const char *token, SparsePythonStorageScheme &scheme)
{
    if (token == nullptr) {
        return false;
    }
    std::string normalized = ToLower(token);
    if (normalized == "csr" || normalized == "row") {
        scheme = SparsePythonStorageScheme::CSR;
        return true;
    }
    if (normalized == "csc" || normalized == "column" || normalized == "col") {
        scheme = SparsePythonStorageScheme::CSC;
        return true;
    }
    if (normalized == "coo" || normalized == "coordinate") {
        scheme = SparsePythonStorageScheme::COO;
        return true;
    }
    return false;
}

// Parse a single writable flag token and update flags
// Returns true if the token was recognized, false otherwise
bool
ParseSingleWritableFlag(const char *token, SparsePythonWritableFlags &flags, bool &hasAll, bool &hasNone)
{
    if (token == nullptr) {
        return false;
    }
    std::string normalized = ToLower(token);
    // Trim whitespace
    normalized.erase(0, normalized.find_first_not_of(" \t"));
    normalized.erase(normalized.find_last_not_of(" \t") + 1);
    
    if (normalized == "values" || normalized == "value" || normalized == "data") {
        flags.values = true;
        return true;
    } else if (normalized == "rhs" || normalized == "b" || normalized == "right-hand-side") {
        flags.rhs = true;
        return true;
    } else if (normalized == "all" || normalized == "true" || normalized == "1") {
        flags.values = true;
        flags.rhs = true;
        hasAll = true;
        return true;
    } else if (normalized == "none" || normalized == "false" || normalized == "0") {
        flags.values = false;
        flags.rhs = false;
        hasNone = true;
        return true;
    }
    
    return false;
}

bool
ParseWritableFlags(const char *token, SparsePythonWritableFlags &flags)
{
    if (token == nullptr) {
        return false;
    }
    std::string normalized = ToLower(token);
    
    flags = SparsePythonWritableFlags(false, false);
    bool hasAll = false;
    bool hasNone = false;
    
    // Parse comma-separated list
    std::istringstream iss(normalized);
    std::string item;
    bool valid = false;
    
    while (std::getline(iss, item, ',')) {
        if (item.empty()) {
            // Skip empty items from multiple commas
            continue;
        }
        
        if (!ParseSingleWritableFlag(item.c_str(), flags, hasAll, hasNone)) {
            // Unknown token
            return false;
        }
        
        valid = true;
        
        // Early exit for complete flags
        if (hasAll || hasNone) {
            break;
        }
    }
    
    return valid;
}

bool
ParseMatrixTypeToken(const char *token, SparsePythonMatrixType &type)
{
    if (token == nullptr) {
        return false;
    }
    std::string normalized = ToLower(token);
    if (normalized == "general") {
        type = SparsePythonMatrixType::GENERAL;
        return true;
    }
    if (normalized == "symmetric_lower" || normalized == "symmetric") {
        type = SparsePythonMatrixType::SYMMETRIC_LOWER;
        return true;
    }
    return false;
}

// Warn about dictionary keys this version does not understand. They used to be
// ignored silently, so an option meant for a newer OpenSees could change the
// meaning of the buffers without any message.
void
WarnUnknownKeys(PyObject *dict, const char *type)
{
    static const char *known[] = {"solver", "scheme", "writable", "matrix_type", "typed"};
    PyObject *key = nullptr;
    PyObject *value = nullptr;
    Py_ssize_t pos = 0;
    while (PyDict_Next(dict, &pos, &key, &value)) {
        const char *name = PyUnicode_Check(key) ? PyUnicode_AsUTF8(key) : nullptr;
        bool found = false;
        for (const char *k : known) {
            if (name != nullptr && strcmp(name, k) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            opserr << "WARNING: system " << type << " - ignoring unknown key '"
                   << (name != nullptr ? name : "?") << "'" << endln;
        }
    }
    PyErr_Clear();
}

PyObject *
EnsureCallable(PyObject *solverObject, const char *solveMethod)
{
    // Ensure GIL is held for Python API calls
    PyGILState_STATE gilState = PyGILState_Ensure();

    if (!PyObject_HasAttrString(solverObject, solveMethod)) {
        opserr << "WARNING: system PythonSparse - solver object missing method '" << solveMethod << "'" << endln;
        PyGILState_Release(gilState);
        return nullptr;
    }
    PyObject *callable = PyObject_GetAttrString(solverObject, solveMethod);
    if (callable == nullptr || !PyCallable_Check(callable)) {
        opserr << "WARNING: system PythonSparse - attribute '" << solveMethod << "' not callable" << endln;
        Py_XDECREF(callable);
        PyGILState_Release(gilState);
        return nullptr;
    }
    PyGILState_Release(gilState);
    return callable; // new reference; caller responsible for DECREF
}

void *
CreateCompressedSOE(PyObject *solverObject, SparsePythonStorageScheme scheme,
                    const SparsePythonWritableFlags &writableFlags,
                    SparsePythonMatrixType theMatrixType,
                    const SparsePythonCallOptions &callOptions)
{
    if (scheme == SparsePythonStorageScheme::COO) {
        opserr << "WARNING: system PythonSparse - COO scheme is not compatible with compressed storage" << endln;
        return nullptr;
    }

    PyObject *callable = EnsureCallable(solverObject, "solve");
    if (callable == nullptr) {
        return nullptr;
    }

    // Acquire GIL before Py_DECREF
    PyGILState_STATE gilState = PyGILState_Ensure();
    Py_DECREF(callable);
    PyGILState_Release(gilState);

    auto *solver = new SparsePythonCompressedLinSolver();
    solver->setPythonCallable(solverObject, "solve");
    solver->setWritableFlags(writableFlags);
    solver->setCallOptions(callOptions);

    auto *soe = new SparsePythonCompressedLinSOE(*solver, scheme, theMatrixType);
    return soe;
}

void *
CreateCOOSOE(PyObject *solverObject, const SparsePythonWritableFlags &writableFlags,
             SparsePythonMatrixType theMatrixType, const SparsePythonCallOptions &callOptions)
{
    PyObject *callable = EnsureCallable(solverObject, "solve");
    if (callable == nullptr) {
        return nullptr;
    }

    // Acquire GIL before Py_DECREF
    PyGILState_STATE gilState = PyGILState_Ensure();
    Py_DECREF(callable);
    PyGILState_Release(gilState);

    auto *solver = new SparsePythonCOOLinSolver();
    solver->setPythonCallable(solverObject, "solve");
    solver->setWritableFlags(writableFlags);
    solver->setCallOptions(callOptions);

    auto *soe = new SparsePythonCOOLinSOE(*solver, theMatrixType);
    return soe;
}

} // namespace

void *
OPS_SparsePythonSolver()
{
    const char *expectedSyntax = "system 'PythonSparse' {'solver': SolverObject, 'scheme': 'CSR'|'CSC'|'COO', 'writable': 'values'|'rhs'|'values,rhs'|'all'|'none', 'matrix_type': 'GENERAL'|'SYMMETRIC_LOWER', 'typed': False|True}";

    OPS_ResetCurrentInputArg(-1);

    const char *type = OPS_GetString();
    if (type == nullptr || !IsSparsePythonType(type)) {
        opserr << "WARNING: system " << (type != nullptr ? type : "PythonSparse")
               << " - unexpected type" << endln;
        opserr << "Expected syntax: " << expectedSyntax << endln;
        return nullptr;
    }

    if (!Py_IsInitialized()) {
        Py_Initialize();
    }

    // Ensure GIL is held for all Python API calls
    PyGILState_STATE gilState = PyGILState_Ensure();

    // Get the dictionary argument
    void *dictPtr = OPS_GetVoidPtr();
    if (dictPtr == nullptr) {
        opserr << "WARNING: system " << type << " - requires a dictionary argument" << endln;
        opserr << "Expected syntax: " << expectedSyntax << endln;
        PyGILState_Release(gilState);
        return nullptr;
    }

    PyObject *dict = static_cast<PyObject *>(dictPtr);
    if (!PyDict_Check(dict)) {
        opserr << "WARNING: system " << type << " - second argument must be a dictionary" << endln;
        opserr << "Expected syntax: " << expectedSyntax << endln;
        PyGILState_Release(gilState);
        return nullptr;
    }

    // Extract 'solver' (required)
    PyObject *solverObj = PyDict_GetItemString(dict, "solver");
    if (solverObj == nullptr) {
        opserr << "WARNING: system " << type << " - dictionary must contain 'solver' key" << endln;
        opserr << "Expected syntax: " << expectedSyntax << endln;
        PyGILState_Release(gilState);
        return nullptr;
    }

    // Extract 'scheme' (optional, defaults to CSR)
    SparsePythonStorageScheme scheme = SparsePythonStorageScheme::CSR;
    PyObject *schemeObj = PyDict_GetItemString(dict, "scheme");
    if (schemeObj != nullptr) {
        if (!PyUnicode_Check(schemeObj)) {
            opserr << "WARNING: system " << type << " - 'scheme' must be a string" << endln;
            PyGILState_Release(gilState);
            return nullptr;
        }
        const char *schemeStr = PyUnicode_AsUTF8(schemeObj);
        if (schemeStr == nullptr || !ParseSchemeToken(schemeStr, scheme)) {
            opserr << "WARNING: system " << type << " - unknown storage scheme '"
                   << (schemeStr != nullptr ? schemeStr : "null") << "' (expected CSR, CSC, or COO)" << endln;
            PyGILState_Release(gilState);
            return nullptr;
        }
    }

    // Extract 'writable' (optional, defaults to read-only)
    // Can be None, a string ('values', 'rhs', 'values,rhs', 'all', 'none'), or a list (['values', 'rhs'])
    SparsePythonWritableFlags writableFlags;  // Default: all read-only
    PyObject *writableObj = PyDict_GetItemString(dict, "writable");
    if (writableObj != nullptr) {
        if (writableObj == Py_None) {
            // None means 'none' (all read-only), which is already the default
            // No need to do anything, writableFlags is already false/false
        } else if (PyUnicode_Check(writableObj)) {
            // String format: 'values,rhs', 'all', 'none', etc.
            const char *writableStr = PyUnicode_AsUTF8(writableObj);
            if (writableStr == nullptr || !ParseWritableFlags(writableStr, writableFlags)) {
                opserr << "WARNING: system " << type << " - invalid writable flags '"
                       << (writableStr != nullptr ? writableStr : "null") << "'" << endln;
                opserr << "Expected: 'values', 'rhs', 'values,rhs', 'all', or 'none'" << endln;
                PyGILState_Release(gilState);
                return nullptr;
            }
        } else if (PyList_Check(writableObj) || PyTuple_Check(writableObj)) {
            // List/tuple format: ['values', 'rhs']
            PyObject *seq = PySequence_Fast(writableObj, "writable must be a sequence");
            if (seq == nullptr) {
                PyGILState_Release(gilState);
                return nullptr;
            }
            
            Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
            writableFlags = SparsePythonWritableFlags(false, false);
            bool valid = false;
            bool hasAll = false;
            bool hasNone = false;
            
            for (Py_ssize_t i = 0; i < len; i++) {
                PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
                if (!PyUnicode_Check(item)) {
                    opserr << "WARNING: system " << type << " - 'writable' list items must be strings" << endln;
                    Py_DECREF(seq);
                    PyGILState_Release(gilState);
                    return nullptr;
                }
                
                const char *itemStr = PyUnicode_AsUTF8(item);
                if (itemStr == nullptr) {
                    Py_DECREF(seq);
                    PyGILState_Release(gilState);
                    return nullptr;
                }
                
                if (!ParseSingleWritableFlag(itemStr, writableFlags, hasAll, hasNone)) {
                    opserr << "WARNING: system " << type << " - unknown writable flag '"
                           << itemStr << "' in list" << endln;
                    opserr << "Expected: 'values', 'rhs', 'all', or 'none'" << endln;
                    Py_DECREF(seq);
                    PyGILState_Release(gilState);
                    return nullptr;
                }
                
                valid = true;
                
                // Early exit for complete flags
                if (hasAll || hasNone) {
                    break;
                }
            }
            
            Py_DECREF(seq);
            if (!valid) {
                opserr << "WARNING: system " << type << " - 'writable' list must contain at least one valid flag" << endln;
                PyGILState_Release(gilState);
                return nullptr;
            }
        } else {
            opserr << "WARNING: system " << type << " - 'writable' must be a string or a list/tuple" << endln;
            PyGILState_Release(gilState);
            return nullptr;
        }
    }

    // Extract 'matrix_type' (optional, defaults to GENERAL). SYMMETRIC_LOWER
    // stores only row >= column and passes matrix_type to solve().
    SparsePythonMatrixType theMatrixType = SparsePythonMatrixType::GENERAL;
    SparsePythonCallOptions callOptions;
    PyObject *matrixTypeObj = PyDict_GetItemString(dict, "matrix_type");
    if (matrixTypeObj != nullptr && matrixTypeObj != Py_None) {
        const char *matrixTypeStr =
            PyUnicode_Check(matrixTypeObj) ? PyUnicode_AsUTF8(matrixTypeObj) : nullptr;
        if (matrixTypeStr == nullptr || !ParseMatrixTypeToken(matrixTypeStr, theMatrixType)) {
            opserr << "WARNING: system " << type << " - unknown matrix_type '"
                   << (matrixTypeStr != nullptr ? matrixTypeStr : "?")
                   << "' (expected GENERAL or SYMMETRIC_LOWER)" << endln;
            PyErr_Clear();
            PyGILState_Release(gilState);
            return nullptr;
        }
        callOptions.passMatrixType = true;
    }

    // Extract 'typed' (optional, defaults to False): memoryviews with format
    // 'i'/'d' instead of raw bytes.
    PyObject *typedObj = PyDict_GetItemString(dict, "typed");
    if (typedObj != nullptr) {
        const int truth = PyObject_IsTrue(typedObj);
        if (truth < 0) {
            PyErr_Clear();
            opserr << "WARNING: system " << type << " - 'typed' must be True or False" << endln;
            PyGILState_Release(gilState);
            return nullptr;
        }
        callOptions.typedBuffers = (truth == 1);
    }

    WarnUnknownKeys(dict, type);

    void *result = nullptr;
    if (scheme == SparsePythonStorageScheme::COO) {
        result = CreateCOOSOE(solverObj, writableFlags, theMatrixType, callOptions);
    } else {
        result = CreateCompressedSOE(solverObj, scheme, writableFlags, theMatrixType, callOptions);
    }

    PyGILState_Release(gilState);
    return result;
}

