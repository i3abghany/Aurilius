#include "Vector2d.h"

Aurilius::Vector2d::Vector2d() : data(new double[2]{0, 0}){}

Aurilius::Vector2d::Vector2d(double x, double y) {
    this->data[0] = x, this->data[1] = y;
}

double Aurilius::Vector2d::dot(const Aurilius::Vector2d &other) {
    double res = this->data[0] * other['x'] + this->data[1] * other['y'];
    return res;
}