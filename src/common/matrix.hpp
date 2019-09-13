#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include <cassert>

#include "utils.hpp"

/**
 *
 * Struct for a matrix.
 *
 */
class Matrix {

public:

    Matrix();
    Matrix(unsigned _rows, unsigned _cols);
    Matrix(unsigned _rows, unsigned _cols, FeatType *_data);
    Matrix(unsigned _rows, unsigned _cols, char *_data);

    unsigned getRows();
    unsigned getCols();
    unsigned getNumElemts();
    FeatType *getData() const;
    size_t getDataSize() const;

    // Get a specific element in the matrix
    FeatType get(unsigned row, unsigned col);

    // Get a full row in the matrix
    // Just returns a pointer to the start of the row (no size information etc)
    FeatType* get(unsigned row);

    void setRows(unsigned _rows);
    void setCols(unsigned _cols);
    void setDims(unsigned _rows, unsigned _cols);
    void setData(FeatType *_data);

    bool empty();

    // Multiply every element by some float
    Matrix operator*(float rhs);
    friend Matrix operator*(float lhs, Matrix& rhs);
    void operator*=(float rhs);

    // Divide every element by some float
    Matrix operator/(float rhs);
    void operator/=(float rhs);

    // Adding some float to every element
    Matrix operator+(float rhs);
    friend Matrix operator+(float lhs, Matrix& rhs);
    void operator+=(float rhs);

    // Adding some float to every element
    Matrix operator-(float rhs);
    void operator-=(float rhs);

    // Elementwise operations on matrices
    Matrix operator*(Matrix& M);
    Matrix operator/(Matrix& M);
    Matrix operator+(Matrix& M);
    Matrix operator-(Matrix& M);

    void operator*=(Matrix& M);
    void operator/=(Matrix& M);
    void operator+=(Matrix& M);
    void operator-=(Matrix& M);

    // Ideally add some sort of numpy style operator for Matmul such
    // as '@' (A @ B equivalent to np.dot(A, B))

    std::string shape();

    std::string str();

private:

    unsigned rows;
    unsigned cols;
    FeatType *data;
};


#endif
