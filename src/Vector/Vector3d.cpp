#include "Vector3d.h"

Aurilius::Vector3d::Vector3d() : data(new double[3]{0, 0, 0}){}

Aurilius::Vector3d::Vector3d(double x, double y, double z) {
    this->data[0] = x, this->data[1] = y, this->data[2] = z;
}

double Aurilius::Vector3d::dot(const Aurilius::Vector3d &b) {
    const double ax = this->data[0], ay = this->data[1], az = this->data[2];
    double res = ax * b['x'] + ay * b['y'] + az * b['z'];
    return res;
}

Aurilius::Vector3d Aurilius::Vector3d::cross(const Aurilius::Vector3d &b) {
    const double ax = this->data[0], ay = this->data[1], az = this->data[2];

    const double x_comp = ay * b['z'] - az * b['y'];
    const double y_comp = ax * b['z'] - az * b['x'];
    const double z_comp = ax * b['y'] - ay * b['x'];

    return Vector3d(x_comp, -y_comp, z_comp);
}