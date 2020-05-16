#include "Activations.h"

double Aurilius::Activations::step(double t) {
    return (t < 0) ? 0.0 : 1.0;
}