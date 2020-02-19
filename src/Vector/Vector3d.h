#pragma once

#include <stdexcept>

namespace Aurilius {

    class Vector3d{
    private:
        double *data;
    public:
        Vector3d(double, double, double);
        Vector3d();

        double dot(const Vector3d&);
        Vector3d cross(const Vector3d&);

        double& operator[](const char idx) const {
            if(idx == 'x') {
                return this->data[0];
            } else if(idx == 'y') {
                return this->data[1];
            } else if(idx == 'z') {
                return this->data[2];
            }
            throw std::runtime_error("Invalid index.\n");
        }

    };
}