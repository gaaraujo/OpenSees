#ifndef SparsePythonCommon_h
#define SparsePythonCommon_h

#include <Python.h>

/**
 * Storage scheme class: acts like an enum with string conversion.
 */
class SparsePythonStorageScheme {
private:
    enum class Value : int {
        CSR = 0,
        CSC = 1,
        COO = 2
    };
    Value value_;

public:
    // Static const members for easy access (defined in .cpp)
    static const SparsePythonStorageScheme CSR;
    static const SparsePythonStorageScheme CSC;
    static const SparsePythonStorageScheme COO;

    // Constructors
    SparsePythonStorageScheme() : value_(Value::CSR) {}
    explicit SparsePythonStorageScheme(Value v) : value_(v) {}
    SparsePythonStorageScheme(int v) : value_(static_cast<Value>(v)) {}

    // Implicit conversion operators for comparisons and assignments
    operator Value() const { return value_; }
    operator int() const { return static_cast<int>(value_); }

    // Comparison operators
    bool operator==(const SparsePythonStorageScheme& other) const {
        return value_ == other.value_;
    }
    bool operator!=(const SparsePythonStorageScheme& other) const {
        return value_ != other.value_;
    }

    // String conversion for Python
    const char* to_str() const;
};

/**
 * Matrix status class: acts like an enum with string conversion.
 */
class SparsePythonMatrixStatus {
private:
    enum class Value : int {
        UNCHANGED = 0,
        COEFFICIENTS_CHANGED = 1,
        STRUCTURE_CHANGED = 2
    };
    Value value_;

public:
    // Static const members for easy access (defined in .cpp)
    static const SparsePythonMatrixStatus UNCHANGED;
    static const SparsePythonMatrixStatus COEFFICIENTS_CHANGED;
    static const SparsePythonMatrixStatus STRUCTURE_CHANGED;

    // Constructors
    SparsePythonMatrixStatus() : value_(Value::UNCHANGED) {}
    explicit SparsePythonMatrixStatus(Value v) : value_(v) {}
    SparsePythonMatrixStatus(int v) : value_(static_cast<Value>(v)) {}

    // Implicit conversion operators for comparisons and assignments
    operator Value() const { return value_; }
    operator int() const { return static_cast<int>(value_); }

    // Comparison operators
    bool operator==(const SparsePythonMatrixStatus& other) const {
        return value_ == other.value_;
    }
    bool operator!=(const SparsePythonMatrixStatus& other) const {
        return value_ != other.value_;
    }

    // String conversion for Python
    const char* to_str() const;
};

/**
 * Flags controlling which buffers are exported as writable to Python solvers.
 * By default, all buffers except 'x' (solution) are read-only for safety.
 * Advanced users can enable writable buffers for in-place modifications.
 */
struct SparsePythonWritableFlags {
    bool values = false;  // Matrix coefficients (values array)
    bool rhs = false;     // Right-hand side vector
    // Note: index_ptr, indices, row, col are always read-only (structural)
    // Note: x (solution) is always writable (output buffer)

    SparsePythonWritableFlags() = default;
    SparsePythonWritableFlags(bool v, bool r) : values(v), rhs(r) {}
};

/**
 * Part of the matrix held by the SOE. SYMMETRIC_LOWER (opt-in) stores only the
 * entries with row >= column; the user asserts that the matrix is symmetric.
 */
class SparsePythonMatrixType {
private:
    enum class Value : int {
        GENERAL = 0,
        SYMMETRIC_LOWER = 1
    };
    Value value_;

public:
    static const SparsePythonMatrixType GENERAL;
    static const SparsePythonMatrixType SYMMETRIC_LOWER;

    SparsePythonMatrixType() : value_(Value::GENERAL) {}
    explicit SparsePythonMatrixType(Value v) : value_(v) {}

    bool operator==(const SparsePythonMatrixType& other) const {
        return value_ == other.value_;
    }
    bool operator!=(const SparsePythonMatrixType& other) const {
        return value_ != other.value_;
    }

    const char* to_str() const;
};

/**
 * Opt-in options that change what is passed to the Python solver. With the
 * defaults the call is identical to OpenSees 3.8.0, so solvers written for
 * that version keep working.
 */
struct SparsePythonCallOptions {
    // Pass 'matrix_type' to solve(). Set when the SOE stores SYMMETRIC_LOWER.
    bool passMatrixType = false;
    // Hand out memoryviews with format 'i' / 'd' instead of raw bytes ('B').
    bool typedBuffers = false;
};

/**
 * Create a 1-D memoryview over count items of size itemsize at data. With
 * typed = false the view has format 'B' (as in OpenSees 3.8.0); with
 * typed = true it is cast to format (e.g. "d" or "i"). Returns a new reference
 * or nullptr with a Python error set. The GIL must be held.
 */
PyObject *SparsePythonMakeView(void *data, Py_ssize_t count, Py_ssize_t itemsize,
                               const char *format, bool writable, bool typed);

/**
 * Convert the object returned by a Python solve() to a status code:
 * None -> 0; True -> 0, False -> -1; an integer (including numpy integers)
 * or a float -> its value. Anything else prints a warning and returns -1.
 * The GIL must be held.
 */
int SparsePythonResultToStatus(PyObject *result, const char *caller);

/**
 * RAII helper that manages reference counting for transient PyObject* instances.
 * Acquires the GIL before calling Py_XDECREF to ensure thread safety.
 */
struct PyObjectHolder {
    PyObjectHolder() = default;
    explicit PyObjectHolder(PyObject *obj) : ptr(obj) {}
    ~PyObjectHolder() {
        // Acquire GIL before calling Py_XDECREF to ensure thread safety
        // Note: We don't set ptr to nullptr here because the object is being destroyed
        if (ptr != nullptr && Py_IsInitialized()) {
            PyGILState_STATE gilState = PyGILState_Ensure();
            Py_XDECREF(ptr);
            PyGILState_Release(gilState);
        }
    }
    PyObject *get() const { return ptr; }
    PyObject *release() {
        PyObject *tmp = ptr;
        ptr = nullptr;
        return tmp;
    }
    void reset(PyObject *obj = nullptr) {
        if (ptr == obj) {
            return;
        }
        // reset() should only be called while GIL is held
        Py_XDECREF(ptr);
        ptr = obj;
    }
  private:
    PyObject *ptr{nullptr};
};

#endif /* SparsePythonCommon_h */
