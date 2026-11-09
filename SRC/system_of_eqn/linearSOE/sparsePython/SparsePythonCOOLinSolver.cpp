#include "SparsePythonCOOLinSolver.h"

#include <Channel.h>
#include <FEM_ObjectBroker.h>
#include <OPS_Globals.h>
#include <elementAPI.h>
#include <Python.h>
#include "SparsePythonCommon.h"

#include <cstring>

SparsePythonCOOLinSolver::SparsePythonCOOLinSolver()
    : LinearSOESolver(SOLVER_TAGS_SparsePythonCOOLinSolver),
      theSOE(nullptr),
      solverObject(nullptr),
      methodName("solve")
{
}

SparsePythonCOOLinSolver::SparsePythonCOOLinSolver(PyObject *callable, const char *method)
    : LinearSOESolver(SOLVER_TAGS_SparsePythonCOOLinSolver),
      theSOE(nullptr),
      solverObject(nullptr),
      methodName(method != nullptr ? method : "solve")
{
    setPythonCallable(callable, methodName.c_str());
}

SparsePythonCOOLinSolver::~SparsePythonCOOLinSolver()
{
    // NOTE: This is generally unsafe to call Python C-API from destructors,
    // as destructors may run after Py_Finalize(). This will be addressed later.
    // For now, we assume the user will call ops.wipe() at the end of their script
    // to ensure resources are released before exiting Python, making it safe to
    // decrement refcounts in destructors.
    if (solverObject != nullptr && Py_IsInitialized()) {
        PyGILState_STATE gilState = PyGILState_Ensure();
        Py_XDECREF(solverObject);
        PyGILState_Release(gilState);
    }
}

int
SparsePythonCOOLinSolver::setLinearSOE(SparsePythonCOOLinSOE &theLinearSOE)
{
    theSOE = &theLinearSOE;
    return 0;
}

int
SparsePythonCOOLinSolver::setPythonCallable(PyObject *callable, const char *method)
{
    if (method != nullptr) {
        methodName = method;
    }

    // Acquire GIL before any Python API calls
    PyGILState_STATE gilState = PyGILState_Ensure();

    if (callable == nullptr) {
        Py_XDECREF(solverObject);
        solverObject = nullptr;
        PyGILState_Release(gilState);
        return 0;
    }

    Py_XINCREF(callable);
    Py_XDECREF(solverObject);
    solverObject = callable;

    PyGILState_Release(gilState);
    return 0;
}


int
SparsePythonCOOLinSolver::solve(void)
{
    if (theSOE == nullptr) {
        opserr << "SparsePythonCOOLinSolver::solve - no associated SOE" << endln;
        return -1;
    }
    if (solverObject == nullptr) {
        opserr << "SparsePythonCOOLinSolver::solve - no Python callable bound" << endln;
        return -2;
    }
    if (theSOE->getNumEqn() == 0) {
        return 0;
    }

    return callPythonSolver();
}

int
SparsePythonCOOLinSolver::setSize(void)
{
    if (theSOE == nullptr) {
        opserr << "SparsePythonCOOLinSolver::setSize - no associated SOE" << endln;
        return -1;
    }
    return 0;
}

int
SparsePythonCOOLinSolver::callPythonSolver()
{
    const std::vector<int> &rowIdx = theSOE->getRowIndices();
    const std::vector<int> &colIdx = theSOE->getColIndices();
    std::vector<double> &values = theSOE->getValues();
    std::vector<double> &rhs = theSOE->getRHS();
    std::vector<double> &sol = theSOE->getSolution();

    if (rowIdx.size() != colIdx.size() || rowIdx.size() != values.size()) {
        opserr << "SparsePythonCOOLinSolver::callPythonSolver - inconsistent COO storage" << endln;
        return -3;
    }

    if (rowIdx.empty()) {
        return 0;
    }

    PyGILState_STATE gilState = PyGILState_Ensure();

    // Lookup the bound solve method; bail if missing or not callable.
    PyObjectHolder method(PyObject_GetAttrString(solverObject, methodName.c_str()));
    if (method.get() == nullptr || !PyCallable_Check(method.get())) {
        opserr << "SparsePythonCOOLinSolver::callPythonSolver - attribute '"
               << methodName.c_str() << "' not callable" << endln;
        PyGILState_Release(gilState);
        return -4;
    }

    // Build memoryviews for COO storage and vectors
    const Py_ssize_t nnz = static_cast<Py_ssize_t>(rowIdx.size());
    const bool typed = callOptions.typedBuffers;
    PyObjectHolder pyRow(SparsePythonMakeView(
        const_cast<int *>(rowIdx.data()), nnz, sizeof(int), "i", false, typed));
    PyObjectHolder pyCol(SparsePythonMakeView(
        const_cast<int *>(colIdx.data()), nnz, sizeof(int), "i", false, typed));
    PyObjectHolder pyValues(SparsePythonMakeView(
        values.data(), nnz, sizeof(double), "d", writableFlags.values, typed));
    PyObjectHolder pyRhs(SparsePythonMakeView(
        rhs.data(), static_cast<Py_ssize_t>(rhs.size()),
        sizeof(double), "d", writableFlags.rhs, typed));
    PyObjectHolder pySol(SparsePythonMakeView(
        sol.data(), static_cast<Py_ssize_t>(sol.size()),
        sizeof(double), "d", true, typed));

    if (pyRow.get() == nullptr || pyCol.get() == nullptr ||
        pyValues.get() == nullptr || pyRhs.get() == nullptr || pySol.get() == nullptr) {
        PyErr_Print();
        opserr << "SparsePythonCOOLinSolver::callPythonSolver - failed to create memory views" << endln;
        PyGILState_Release(gilState);
        return -5;
    }

    // Scalar metadata handed to the Python solver
    PyObjectHolder matrixStatus(
        PyUnicode_FromString(theSOE->getMatrixStatus().to_str()));
    PyObjectHolder storageScheme(
        PyUnicode_FromString(SparsePythonStorageScheme::COO.to_str()));

    PyObjectHolder kwargs(PyDict_New());
    const char *rowKeyword = "row";
    const char *colKeyword = "col";
    const char *valuesKeyword = "values";
    const char *rhsKeyword = "rhs";
    const char *xKeyword = "x";
    const char *matrixStatusKeyword = "matrix_status";
    const char *storageSchemeKeyword = "storage_scheme";
    const char *numEqnKeyword = "num_eqn";
    const char *nnzKeyword = "nnz";

    if (kwargs.get() == nullptr ||
        PyDict_SetItemString(kwargs.get(), rowKeyword, pyRow.get()) != 0 ||
        PyDict_SetItemString(kwargs.get(), colKeyword, pyCol.get()) != 0 ||
        PyDict_SetItemString(kwargs.get(), valuesKeyword, pyValues.get()) != 0 ||
        PyDict_SetItemString(kwargs.get(), rhsKeyword, pyRhs.get()) != 0 ||
        PyDict_SetItemString(kwargs.get(), xKeyword, pySol.get()) != 0 ||
        matrixStatus.get() == nullptr ||
        storageScheme.get() == nullptr ||
        PyDict_SetItemString(kwargs.get(), matrixStatusKeyword, matrixStatus.get()) != 0 ||
        PyDict_SetItemString(kwargs.get(), storageSchemeKeyword, storageScheme.get()) != 0) {
        opserr << "SparsePythonCOOLinSolver::callPythonSolver - failed to populate kwargs" << endln;
        PyGILState_Release(gilState);
        return -6;
    }

    PyObjectHolder numEqn(PyLong_FromLong(theSOE->getNumEqn()));
    PyObjectHolder nnzValue(PyLong_FromSize_t(theSOE->getNNZ()));

    if (numEqn.get() == nullptr || nnzValue.get() == nullptr ||
        PyDict_SetItemString(kwargs.get(), numEqnKeyword, numEqn.get()) != 0 ||
        PyDict_SetItemString(kwargs.get(), nnzKeyword, nnzValue.get()) != 0) {
        opserr << "SparsePythonCOOLinSolver::callPythonSolver - failed to set metadata kwargs" << endln;
        PyGILState_Release(gilState);
        return -7;
    }

    // Opt-in keys are added only when requested, so solve() methods with an
    // explicit signature written for OpenSees 3.8.0 keep working.
    if (callOptions.passMatrixType) {
        PyObjectHolder pyMatrixType(PyUnicode_FromString(theSOE->getMatrixType().to_str()));
        if (pyMatrixType.get() == nullptr ||
            PyDict_SetItemString(kwargs.get(), "matrix_type", pyMatrixType.get()) != 0) {
            opserr << "SparsePythonCOOLinSolver::callPythonSolver - failed to set matrix_type\n";
            PyGILState_Release(gilState);
            return -7;
        }
    }

    PyObject *args = PyTuple_New(0);
    if (args == nullptr) {
        opserr << "SparsePythonCOOLinSolver::callPythonSolver - failed to create args tuple" << endln;
        PyGILState_Release(gilState);
        return -8;
    }
    PyObjectHolder argsHolder(args);

    // PyObject_Call must be made with GIL held. CuPy/NumPy will release the GIL
    // internally during their native GPU/CPU computations, so we don't need to
    // do anything special here.
    PyObjectHolder result(PyObject_Call(method.get(), argsHolder.get(), kwargs.get()));
    if (result.get() == nullptr) {
        opserr << "SparsePythonCOOLinSolver::callPythonSolver - Python call failed\n";
        PyErr_Print();
        PyGILState_Release(gilState);
        return -9;
    }

    const int status = SparsePythonResultToStatus(result.get(), "SparsePythonCOOLinSolver::solve");

    PyGILState_Release(gilState);
    return status;
}

int
SparsePythonCOOLinSolver::sendSelf(int, Channel &)
{
    return 0;
}

int
SparsePythonCOOLinSolver::recvSelf(int, Channel &, FEM_ObjectBroker &)
{
    return 0;
}
