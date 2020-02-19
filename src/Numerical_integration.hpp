#pragma once
#include <limits>
#include <cmath>

namespace Aurilius {
    const double EPS = std::numeric_limits<double>::epsilon() * 1e5;

    template<typename Callable>
    double trapezoidal(const Callable &f, const double a, const double b, const std::size_t order = 1000) {
        const double delta = (b - a) / (1.0 * order);
        double curr_x = a;
        double ans = f(curr_x);
        curr_x += delta;

        const std::size_t n_threads = 8;

        #pragma omp parallel for num_threads(n_threads) reduction(+: ans)
        for (int i = 0; i < order; i++) {
            ans += 2 * f(curr_x);
            curr_x += delta;
        }

        ans += f(curr_x + delta);
        ans *= (delta / 2.0);
        return ans;
    }

    template<typename Callable>
    double simpson(const Callable &f, const double a, const double b, const std::size_t order = 1000) {
        const double delta = (b - a) / (1.0 * order);
        double curr_x = a;
        double ans = f(curr_x);
        curr_x += delta;
        bool alt = true;
        for (; std::fabs(curr_x - b) > EPS; curr_x += delta) {
            if (alt) ans += 4 * f(curr_x);
            else ans += 2 * f(curr_x);

            alt = !alt;
        }
        ans += f(curr_x + delta);
        ans *= (delta / 3.0);
        return ans;
    }
}