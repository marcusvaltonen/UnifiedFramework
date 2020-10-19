#include <assert.h>
#include <math.h>
#include <float.h>  // DBL_MAX
#include <vector>

inline void getMaximizingVector(
    std::vector<double> &a,
    std::vector<double> &b,
    std::vector<double> &sx,
    std::vector<double> &sigma
);
inline int isArraySortedDescend(const std::vector<double> &vect);
inline int isArraySortedAscend(const std::vector<double> &vect);
inline double costFunction(int N, double *a, double *b, double *sx, double z);
inline void findLocalMaximaMinima(
    const std::vector<double> &arr,
    std::vector<int> &mx,
    std::vector<int> &mn
);

#define MAX(a,b) (((a)>(b))?(a):(b))

#ifdef MATLAB_MEX_FILE
#include "mex.h"  // NOLINT [build/include_subdir]
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("get_maximizing_vector:nrhs", "Three inputs required.");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("get_maximizing_vector:nlhs", "One output required.");
    }
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ||
        !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) ||
        !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2])) {
        mexErrMsgIdAndTxt("get_maximizing_vector:notDouble", "Input data must be type double.");
    }
    int N = mxGetNumberOfElements(prhs[0]);

    if (N != mxGetNumberOfElements(prhs[1]) || N != mxGetNumberOfElements(prhs[2])) {
        mexErrMsgIdAndTxt("get_maximizing_vector:length", "All input vectors must be of same size.");
    }

    // Other
    std::vector<double> a, b, sx, sigma;
    a.reserve(N);
    b.reserve(N);
    sx.reserve(N);
    sigma.reserve(N);

    // Fill vectors
    double *a_ptr = (double *) mxGetData(prhs[0]);
    double *b_ptr = (double *) mxGetData(prhs[1]);
    double *sx_ptr = (double *) mxGetData(prhs[2]);
    for (int i = 0; i < N; i++) {
        a.push_back(a_ptr[i]);
        b.push_back(b_ptr[i]);
        sx.push_back(sx_ptr[i]);
    }

    if (!isArraySortedAscend(a)) {
        mexErrMsgIdAndTxt("get_maximizing_vector:vector_not_ascending_input1",
                          "First input vector must be non-decreasing.");
    }
    if (!isArraySortedAscend(b)) {
        mexErrMsgIdAndTxt("get_maximizing_vector:vector_not_ascending_input2",
                          "Second input vector must be non-decreasing.");
    }
    if (!isArraySortedDescend(sx)) {
        mexErrMsgIdAndTxt("get_maximizing_vector:vector_not_decending_input3",
                          "Third input vector must be non-increasing.");
    }

    // Call core function
    getMaximizingVector(a, b, sx, sigma);

    // Wrap it up to Matlab compatible output
    plhs[0] = mxCreateDoubleMatrix(N, 1, mxREAL);
    double* zr = mxGetPr(plhs[0]);
    for (int i = 0; i < N; i++) {
        zr[i] = sigma[i];
    }
}
#endif  // MATLAB_MEX_FILE

inline void getMaximizingVector(std::vector<double> &a, std::vector<double> &b,
                                std::vector<double> &sx, std::vector<double> &sigma) {
    int N = a.size();
    assert(N == b.size());
    assert(N == sx.size());

    // Compute si (which will be namned sigma - avoid copy)
    for (int i = 0; i < N; i++) {
        sigma.push_back(a[i] + MAX(sqrt(b[i]), sx[i]));
    }

    int length_interval, length_subinterval;
    double sumsx;
    std::vector<int> mx, mn;

    while (!isArraySortedDescend(sigma)) {
        mx.clear();
        mn.clear();
        findLocalMaximaMinima(sigma, mx, mn);

        int nbr_intervals = mx.size();
        assert(nbr_intervals == mn.size());
        for (int i = 0; i < nbr_intervals; i++) {
            length_interval = mx[i] - mn[i] + 1;

            // Precompute sum sx(idx)
            sumsx = 0;
            for (int j = 0; j < length_interval; j++) {
                sumsx += sx[mn[i]+j];
            }

            double potential_points[2 * length_interval];
            for (int j = 0; j < length_interval; j++) {
                // Get the potential points coming from derivatives equal to zero
                potential_points[j] = 0;
                for (int k = 0; k < length_interval-j; k++) {
                    potential_points[j] += a[mn[i]+k];
                }
                length_subinterval = length_interval - j;
                potential_points[j] = 1.0 / length_subinterval * (sumsx + potential_points[j]);
                // Get the potential points in between intervals
                potential_points[j+length_interval] = a[mn[i]+j] + sqrt(b[mn[i]+j]);
            }

            // Pick the best from the list of potential points
            // TODO: Not all function evals are necessary, since it is convex.
            double best_val = DBL_MAX;
            double best_pt = -1.0;
            double this_val;

            for (int j = 0; j < 2 * length_interval; j++) {
                this_val = costFunction(length_interval, &a[mn[i]], &b[mn[i]], &sx[mn[i]], potential_points[j]);
                if (this_val < best_val) {
                    best_val = this_val;
                    best_pt = potential_points[j];
                }
            }

            // Set subinterval to constant best_pt
            for (int j = 0; j < length_interval; j++) {
                sigma[mn[i]+j] = best_pt;
            }
        }
    }
}

inline void findLocalMaximaMinima(const std::vector<double> &arr, std::vector<int> &mx, std::vector<int> &mn)
{
    int N = arr.size();

    // Checking whether the first point is local min (otherwise not interesting)
    if (arr[0] < arr[1]) {
        mn.push_back(0);
    }

    // Iterating over all points to check
    // local maxima and local minima
    int flat = 0;
    double dir = arr[1] - arr[0];
    double prev_dir = 0;
    for (int i = 1; i < N - 1; i++) {
        if (arr[i - 1] == arr[i]) {
            flat++;
        } else {
            flat = 0;
            // If positive then increasing
            prev_dir = arr[i] - arr[i-1];
        }
        dir = arr[i+1] - arr[i];

        // Condition for local minima
        if ((arr[i - 1] >= arr[i]) && (arr[i] < arr[i + 1]) && dir > 0 && prev_dir <= 0) {
            mn.push_back(i-flat);
        // Condition for local maxima
        } else if ((arr[i - 1] <= arr[i]) && (arr[i] > arr[i + 1]) && dir < 0 && prev_dir > 0) {
            mx.push_back(i);
        }
    }

    // Checking whether the last point is local maxima
    if (arr[N - 1] > arr[N - 2]) {
        mx.push_back(N - 1);
    }

    // They should be the same length
    if (mx.size() < mn.size()) {
        if (arr[mn.back()] >= arr[N - 1]) {
            mn.pop_back();
        } else {
            mx.push_back(N - 1);
        }
    }
}

inline double costFunction(int N, double *a, double *b, double *sx, double z) {
    double val = 0;
    for (int i = 0; i < N; i++) {
        // The constants have been excluded
        val += MAX(std::pow(MAX(z - a[i], 0), 2) - b[i], 0) - 2 * sx[i] * z;
    }
    return val;
}

inline int isArraySortedDescend(const std::vector<double> &vect) {
    int is_descending = 1;
    int i = 0;
    while ((is_descending == 1) && i < vect.size() - 1) {
        if (vect[i] < vect[i+1]) {
            is_descending = 0;
        }
        i++;
    }
    return is_descending;
}

inline int isArraySortedAscend(const std::vector<double> &vect) {
    int is_ascending = 1;
    int i = 0;

    while ((is_ascending == 1) && i < vect.size() - 1) {
        if (vect[i] > vect[i+1]) {
            is_ascending = 0;
        }
        i++;
    }
    return is_ascending;
}
