#include "SparsePythonCommon.h"

#include <OPS_Globals.h>

// Define static const members for SparsePythonStorageScheme
const SparsePythonStorageScheme SparsePythonStorageScheme::CSR(SparsePythonStorageScheme::Value::CSR);
const SparsePythonStorageScheme SparsePythonStorageScheme::CSC(SparsePythonStorageScheme::Value::CSC);
const SparsePythonStorageScheme SparsePythonStorageScheme::COO(SparsePythonStorageScheme::Value::COO);

const char* SparsePythonStorageScheme::to_str() const {
    switch (value_) {
        case Value::CSR:
            return "CSR";
        case Value::CSC:
            return "CSC";
        case Value::COO:
            return "COO";
        default:
            return "CSR";  // fallback
    }
}

// Define static const members for SparsePythonMatrixStatus
const SparsePythonMatrixStatus SparsePythonMatrixStatus::UNCHANGED(SparsePythonMatrixStatus::Value::UNCHANGED);
const SparsePythonMatrixStatus SparsePythonMatrixStatus::COEFFICIENTS_CHANGED(SparsePythonMatrixStatus::Value::COEFFICIENTS_CHANGED);
const SparsePythonMatrixStatus SparsePythonMatrixStatus::STRUCTURE_CHANGED(SparsePythonMatrixStatus::Value::STRUCTURE_CHANGED);

const char* SparsePythonMatrixStatus::to_str() const {
    switch (value_) {
        case Value::UNCHANGED:
            return "UNCHANGED";
        case Value::COEFFICIENTS_CHANGED:
            return "COEFFICIENTS_CHANGED";
        case Value::STRUCTURE_CHANGED:
            return "STRUCTURE_CHANGED";
        default:
            return "UNCHANGED";  // fallback
    }
}


const SparsePythonMatrixType SparsePythonMatrixType::GENERAL(SparsePythonMatrixType::Value::GENERAL);
const SparsePythonMatrixType SparsePythonMatrixType::SYMMETRIC_LOWER(SparsePythonMatrixType::Value::SYMMETRIC_LOWER);

const char* SparsePythonMatrixType::to_str() const {
    switch (value_) {
        case Value::SYMMETRIC_LOWER:
            return "SYMMETRIC_LOWER";
        case Value::GENERAL:
        default:
            return "GENERAL";
    }
}

PyObject *
SparsePythonMakeView(void *data, Py_ssize_t count, Py_ssize_t itemsize,
                     const char *format, bool writable, bool typed)
{
    PyObject *bytesView = PyMemoryView_FromMemory(
        static_cast<char *>(data), count * itemsize, writable ? PyBUF_WRITE : PyBUF_READ);
    if (bytesView == nullptr || !typed) {
        return bytesView;
    }
    PyObject *typedView = PyObject_CallMethod(bytesView, "cast", "s", format);
    Py_DECREF(bytesView);
    return typedView;
}

int
SparsePythonResultToStatus(PyObject *result, const char *caller)
{
    if (result == Py_None) {
        return 0;
    }
    if (PyBool_Check(result)) {
        return result == Py_True ? 0 : -1;
    }
    if (PyFloat_Check(result)) {
        return static_cast<int>(PyFloat_AsDouble(result));
    }
    // PyNumber_Index accepts Python ints and anything with __index__ (numpy integers).
    PyObject *index = PyNumber_Index(result);
    if (index != nullptr) {
        const long value = PyLong_AsLong(index);
        Py_DECREF(index);
        if (value == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return -1;
        }
        return static_cast<int>(value);
    }
    PyErr_Clear();
    PyObject *typeName = PyObject_GetAttrString(reinterpret_cast<PyObject *>(Py_TYPE(result)), "__name__");
    const char *name = (typeName != nullptr && PyUnicode_Check(typeName)) ? PyUnicode_AsUTF8(typeName) : nullptr;
    opserr << "WARNING: " << caller << " - solve() returned an object of type '"
           << (name != nullptr ? name : "?")
           << "'; expected None or an integer status. Treating it as a failure.\n";
    Py_XDECREF(typeName);
    PyErr_Clear();
    return -1;
}
