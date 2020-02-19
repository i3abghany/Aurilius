#pragma once

#include <stdexcept>

namespace Aurilius {

class Vector2d{
private:
    double *data;
public:
    Vector2d(double, double);
    Vector2d();

    double dot(const Vector2d&);

    double& operator[](const char idx) const {
        if(idx == 'x') {
            return this->data[0];
        } else if(idx == 'y') {
            return this->data[1];
        }
        throw std::runtime_error("Invalid index.\n");
    }

};

}