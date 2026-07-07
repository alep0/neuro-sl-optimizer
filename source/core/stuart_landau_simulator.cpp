/**
 * @file stuart_landau_simulator.cpp
 * @brief C++ accelerated Stuart-Landau neural network simulator with pybind11 bindings.
 *
 * Implements a delayed-coupled Stuart-Landau oscillator network on a structural
 * connectome. Supports two connectivity layers (C1/C2), additive Gaussian noise,
 * and OpenMP parallelism for the coupling loop.
 *
 * PERFORMANCE NOTES (Eigen port)
 * -------------------------------
 * This version replaces the original std::vector<std::vector<T>> storage with
 * Eigen matrices/vectors and sparse structures. The main wins are:
 *
 *   1. Connectivity is stored as Eigen::SparseMatrix<> (row-major, CSR-like),
 *      built ONCE in set_connectivity() from the dense C1/C2 weight matrices.
 *      Real structural connectomes are usually sparse (many zero weights).
 *      The original code looped over all N columns per row every single
 *      timestep and branched on "if (kC1[n][j] != 0.0)" to skip absent edges.
 *      With the sparse representation, the per-timestep coupling loop visits
 *      ONLY the actual edges (via Eigen's InnerIterator) -- no wasted
 *      iterations, no branch on every candidate edge, and it scales with the
 *      number of connections rather than N^2. For a connectome with e.g. 10%
 *      density this alone removes ~90% of the inner-loop work every step.
 *   2. The per-edge delay index is stored in a companion sparse matrix built
 *      with the identical sparsity pattern, so the two can be walked in
 *      lock-step with two InnerIterators -- no re-deriving indices, no
 *      redundant zero checks.
 *   3. The per-timestep local Stuart-Landau update, the noise+coupling state
 *      update, and the output write are expressed as Eigen array/vector
 *      expressions, which are compiled down to SIMD (SSE/AVX) instructions
 *      instead of scalar loops.
 *   4. The history-buffer shift (previously an O(N * max_history) loop run
 *      every step) is now a single Eigen block assignment
 *      (Z.leftCols(max_history_-1) = Z.rightCols(max_history_-1)), which Eigen
 *      lowers to a vectorized block copy.
 *   5. Eigen::Map is used to interpret incoming numpy buffers directly as
 *      Eigen matrices when building the sparse structures, removing the
 *      manual element-by-element copy loops.
 *
 * The remaining gather, Z(j, di), is still a scatter/gather by nature (the
 * delay index varies per edge), but it now only happens once per real edge
 * instead of once per (n, j) candidate pair, and Z itself is a single
 * contiguous Eigen allocation rather than N separately-allocated rows.
 *
 * Build via setup.py:
 *   python setup.py build_ext --inplace
 *
 * NOTE: Eigen is header-only but you must add its include path, e.g. in
 * setup.py's Extension(..., include_dirs=[..., "/usr/include/eigen3"]) or
 * via `pkg-config --cflags eigen3`. Recommended flags: -O3 -march=native
 * -DNDEBUG -fopenmp (or your platform's OpenMP equivalent).
 *
 * Python interface (module name: stuart_landau_simulator):
 *   sim = StuartLandauSimulator(N, max_history, dt, dt_save, tmax, t_prev,
 *                               sig_noise, mean_delay)
 *   sim.set_connectivity(C1, Delays1 [, C2, Delays2])
 *   trajectory = sim.simulate(K, a, f)   # returns np.ndarray (N, n_save)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <complex>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using Complex = std::complex<double>;

// Row-major so that, for a fixed row n, all entries (varying j) are
// contiguous in memory -- matches numpy's default C-contiguous layout too,
// so incoming buffers can be wrapped with Eigen::Map with no copy loop.
using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXiR = Eigen::Matrix<int,    Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Row-major to match numpy's default (C-contiguous) memory layout, so the
// output array can be written via Eigen expressions with a zero-copy Map.
using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Row-major sparse (CSR-like) storage for the connectivity/delay matrices.
// Row-major means InnerIterator over row n walks that row's nonzeros
// contiguously -- exactly the access pattern the coupling loop needs.
using SpMatXd = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using SpMatXi = Eigen::SparseMatrix<int,    Eigen::RowMajor>;

// ---------------------------------------------------------------------------
// Helper: format a log message with a consistent prefix
// ---------------------------------------------------------------------------
static void log_info(const std::string& msg) {
    std::cout << "[StuartLandau] " << msg << std::endl;
}

// ---------------------------------------------------------------------------
class StuartLandauSimulator {
private:
    // Network dimensions
    int N_;
    int max_history_;

    // Time parameters
    double dt_;
    double dt_save_;
    double tmax_;
    double t_prev_;

    // Physical parameters
    double sig_noise_;
    double mean_delay_;   // formerly MD; controls whether history buffer shifts

    // Connectivity (weights) and delay matrices, stored sparse: only actual
    // edges (C != 0) are kept, and the delay matrix is built with the exact
    // same (row, col) pattern so the two can be walked in lock-step every
    // timestep with no branch and no wasted iteration over absent edges.
    SpMatXd C1_sp_;
    SpMatXd C2_sp_;
    SpMatXi Delays1_sp_;
    SpMatXi Delays2_sp_;
    bool use_C2_;

    // Random number generation
    std::mt19937_64 rng_;
    std::normal_distribution<double> normal_dist_;

    // -----------------------------------------------------------------------
    // Internal: build matched-pattern sparse weight/delay matrices from a
    // pair of dense numpy buffers (C-contiguous, row-major -- matches numpy
    // default), wrapped with Eigen::Map so no manual element copy loop is
    // needed for the initial dense read. Only entries where the weight is
    // nonzero are kept, since those are the only ones the simulation ever
    // uses -- this is what lets the per-timestep loop skip absent edges.
    // -----------------------------------------------------------------------
    void build_sparse(py::array_t<double>& C_arr,
                       py::array_t<int>&    D_arr,
                       SpMatXd&             C_sp,
                       SpMatXi&             D_sp)
    {
        auto c_buf = C_arr.request();
        auto d_buf = D_arr.request();
        Eigen::Map<const MatrixXdR> C_dense(static_cast<double*>(c_buf.ptr), N_, N_);
        Eigen::Map<const MatrixXiR> D_dense(static_cast<int*>(d_buf.ptr),   N_, N_);

        std::vector<Eigen::Triplet<double>> tw;
        std::vector<Eigen::Triplet<int>>    td;
        tw.reserve(static_cast<size_t>(N_) * 4);
        td.reserve(static_cast<size_t>(N_) * 4);

        for (int i = 0; i < N_; ++i) {
            for (int j = 0; j < N_; ++j) {
                const double w = C_dense(i, j);
                if (w != 0.0) {
                    tw.emplace_back(i, j, w);
                    td.emplace_back(i, j, D_dense(i, j));
                }
            }
        }

        C_sp.resize(N_, N_);
        D_sp.resize(N_, N_);
        C_sp.setFromTriplets(tw.begin(), tw.end());
        D_sp.setFromTriplets(td.begin(), td.end());
        C_sp.makeCompressed();
        D_sp.makeCompressed();
    }

public:
    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------
    StuartLandauSimulator(int    N,
                          int    max_history,
                          double dt,
                          double dt_save,
                          double tmax,
                          double t_prev,
                          double sig_noise,
                          double mean_delay)
        : N_(N),
          max_history_(max_history),
          dt_(dt),
          dt_save_(dt_save),
          tmax_(tmax),
          t_prev_(t_prev),
          sig_noise_(sig_noise),
          mean_delay_(mean_delay),
          use_C2_(false),
          rng_(std::random_device{}()),
          normal_dist_(0.0, 1.0)
    {
        if (N_ <= 0)
            throw std::invalid_argument("N must be positive.");
        if (max_history_ <= 0)
            throw std::invalid_argument("max_history must be positive.");
        if (dt_ <= 0.0)
            throw std::invalid_argument("dt must be positive.");
        if (dt_save_ < dt_)
            throw std::invalid_argument("dt_save must be >= dt.");
        if (tmax_ <= 0.0)
            throw std::invalid_argument("tmax must be positive.");

        log_info("Simulator constructed. N=" + std::to_string(N_) +
                 ", max_history=" + std::to_string(max_history_) +
                 ", dt=" + std::to_string(dt_) +
                 ", tmax=" + std::to_string(tmax_));
    }

    // -----------------------------------------------------------------------
    // Set connectivity (C1 mandatory; C2 optional)
    // -----------------------------------------------------------------------
    void set_connectivity(py::array_t<double> C1_arr,
                          py::array_t<int>    Delays1_arr,
                          py::object          C2_obj     = py::none(),
                          py::object          Delays2_obj = py::none())
    {
        build_sparse(C1_arr, Delays1_arr, C1_sp_, Delays1_sp_);

        if (!C2_obj.is_none()) {
            if (Delays2_obj.is_none())
                throw std::invalid_argument(
                    "Delays2 must be provided when C2 is provided.");

            use_C2_ = true;

            auto C2_arr      = C2_obj.cast<py::array_t<double>>();
            auto Delays2_arr = Delays2_obj.cast<py::array_t<int>>();
            build_sparse(C2_arr, Delays2_arr, C2_sp_, Delays2_sp_);

            log_info("Two-layer connectivity loaded (C1 + C2). nnz(C1)=" +
                     std::to_string(C1_sp_.nonZeros()) +
                     ", nnz(C2)=" + std::to_string(C2_sp_.nonZeros()));
        } else {
            log_info("Single-layer connectivity loaded (C1 only). nnz(C1)=" +
                     std::to_string(C1_sp_.nonZeros()));
        }
    }

    // -----------------------------------------------------------------------
    // Main simulation
    // -----------------------------------------------------------------------
    py::array_t<double> simulate(double              K,
                                 double              a,
                                 py::array_t<double> f_arr)
    {
        // Validate frequency array length
        auto f_buf = f_arr.request();
        if (f_buf.shape[0] != N_)
            throw std::invalid_argument(
                "f must have length N=" + std::to_string(N_) +
                ", got " + std::to_string(f_buf.shape[0]));
        double* f_ptr = static_cast<double*>(f_buf.ptr);

        const int n_steps       = static_cast<int>((tmax_ + t_prev_) / dt_);
        const int n_save        = static_cast<int>(tmax_ / dt_save_);
        const int save_interval = static_cast<int>(dt_save_ / dt_);
        const int log_interval  = std::max(1, n_steps / 10);

        log_info("Simulation starting: n_steps=" + std::to_string(n_steps) +
                 ", n_save=" + std::to_string(n_save) +
                 ", K=" + std::to_string(K) +
                 ", a=" + std::to_string(a));

        // ------------------------------------------------------------------
        // Pre-compute constants
        // ------------------------------------------------------------------
        const Complex a_complex(a, 0.0);

        Eigen::VectorXcd iomega(N_);
        for (int i = 0; i < N_; ++i)
            iomega(i) = Complex(0.0, 2.0 * M_PI * f_ptr[i]);

        // a + i*omega is constant across the whole simulation -- precompute
        // it once instead of recomputing it every timestep.
        const Eigen::VectorXcd aiomega =
            Eigen::VectorXcd::Constant(N_, a_complex) + iomega;

        const double Kdt   = K * dt_;
        const double dsig  = std::sqrt(dt_) * sig_noise_;

        // ------------------------------------------------------------------
        // Initialise state history with small noise
        // ------------------------------------------------------------------
        // Column-major (Eigen default): Z.col(k) is contiguous, which
        // matches the access pattern Z(j, di) for varying j at fixed di
        // in the coupling loop below.
        Eigen::MatrixXcd Z(N_, max_history_);
        for (int i = 0; i < N_; ++i)
            for (int j = 0; j < max_history_; ++j)
                Z(i, j) = Complex(dt_ * normal_dist_(rng_),
                                  dt_ * normal_dist_(rng_));

        // ------------------------------------------------------------------
        // Allocate output array and wrap it (zero-copy) as a row-major
        // Eigen matrix matching numpy's default C-contiguous layout.
        // ------------------------------------------------------------------
        auto result     = py::array_t<double>({N_, n_save});
        auto result_buf = result.request();
        double* out_ptr = static_cast<double*>(result_buf.ptr);
        Eigen::Map<MatrixXdRowMajor> out_map(out_ptr, N_, n_save);

        // Temporary per-step vectors
        Eigen::VectorXcd Znow(N_), dz(N_), noise(N_),
                          coupling1(N_), coupling2(N_);
        Eigen::ArrayXd abs_sq(N_);

        int    nt = 0;
        double t  = dt_;

        // ------------------------------------------------------------------
        // Main loop
        // ------------------------------------------------------------------
        for (int step = 0; step < n_steps; ++step, t += dt_) {

            if (step % log_interval == 0) {
                double pct = 100.0 * t / tmax_;
                log_info("Progress: " + std::to_string(pct) +
                         "% (t=" + std::to_string(t) + "s)");
            }

            // Snapshot current state
            Znow = Z.col(max_history_ - 1);

            // Stuart-Landau local dynamics (vectorized via Eigen arrays)
            abs_sq = Znow.array().abs2();
            dz = ((Znow.array() * (aiomega.array() - abs_sq.cast<Complex>()))
                  * dt_).matrix();

            // Delayed coupling. The delay index varies per edge, so this
            // cannot be expressed as a dense matrix-vector product -- but
            // walking the sparse matrices means we only ever touch edges
            // that actually exist (no wasted iterations, no branch on a
            // zero weight), and weight/delay are read in lock-step from
            // matched-pattern sparse matrices built once in set_connectivity.
#pragma omp parallel for schedule(static)
            for (int n = 0; n < N_; ++n) {
                Complex s1(0.0, 0.0), s2(0.0, 0.0);

                {
                    SpMatXd::InnerIterator itW(C1_sp_, n);
                    SpMatXi::InnerIterator itD(Delays1_sp_, n);
                    for (; itW; ++itW, ++itD) {
                        const int j     = itW.col();
                        const int di    = itD.value() - 1;
                        const double k1 = itW.value() * Kdt;
                        s1 += k1 * (Z(j, di) - Znow(n));
                    }
                }

                if (use_C2_) {
                    SpMatXd::InnerIterator itW(C2_sp_, n);
                    SpMatXi::InnerIterator itD(Delays2_sp_, n);
                    for (; itW; ++itW, ++itD) {
                        const int j     = itW.col();
                        const int di    = itD.value() - 1;
                        const double k2 = itW.value() * Kdt;
                        s2 += k2 * (Z(j, di) - Znow(n));
                    }
                }

                coupling1(n) = s1;
                coupling2(n) = s2;
            }

            // Shift history buffer -- a single vectorized block assignment
            // instead of an O(N * max_history) element-wise loop.
            if (mean_delay_ > 0.0) {
                Z.leftCols(max_history_ - 1) = Z.rightCols(max_history_ - 1);
            }

            // Generate noise (inherently serial due to the RNG stream)
            for (int i = 0; i < N_; ++i) {
                noise(i) = Complex(dsig * normal_dist_(rng_),
                                    dsig * normal_dist_(rng_));
            }

            // Add noise and update state (vectorized)
            Eigen::VectorXcd coupling_total = coupling1;
            if (use_C2_)
                coupling_total += coupling2;
            Z.col(max_history_ - 1) = Znow + dz + noise + coupling_total;

            // Record output
            if (t > t_prev_ && (step % save_interval == 0) && (nt < n_save)) {
                out_map.col(nt) = Z.col(max_history_ - 1).real();
                ++nt;
            }
        }

        log_info("Simulation complete. Saved " + std::to_string(nt) +
                 " time points.");
        return result;
    }
};

// ---------------------------------------------------------------------------
// pybind11 module definition
// ---------------------------------------------------------------------------
#ifndef MODULE_NAME
#define MODULE_NAME stuart_landau_simulator
#endif
PYBIND11_MODULE(MODULE_NAME, m) {
    m.doc() = R"pbdoc(
        C++ accelerated Stuart-Landau neural network simulator (Eigen-backed).

        Provides ``StuartLandauSimulator``, a delayed-coupled oscillator
        network that models resting-state neural dynamics on an empirical
        structural connectome.
    )pbdoc";

    py::class_<StuartLandauSimulator>(m, "StuartLandauSimulator",
        R"pbdoc(
        Delayed-coupled Stuart-Landau oscillator network.

        Parameters
        ----------
        N : int
            Number of oscillators (network nodes).
        max_history : int
            History buffer length in time-steps (>= max delay + 1).
        dt : float
            Integration time step (s).
        dt_save : float
            Output sampling interval (s). Must satisfy dt_save >= dt.
        tmax : float
            Total simulation time (s).
        t_prev : float
            Warm-up time before recording begins (s).
        sig_noise : float
            Amplitude of complex Gaussian noise.
        mean_delay : float
            Mean axonal delay (s). If 0 the history buffer is not shifted.
        )pbdoc")

        .def(py::init<int, int, double, double, double, double, double, double>(),
             py::arg("N"),
             py::arg("max_history"),
             py::arg("dt"),
             py::arg("dt_save"),
             py::arg("tmax"),
             py::arg("t_prev"),
             py::arg("sig_noise"),
             py::arg("mean_delay"))

        .def("set_connectivity",
             &StuartLandauSimulator::set_connectivity,
             py::arg("C1"),
             py::arg("Delays1"),
             py::arg("C2")      = py::none(),
             py::arg("Delays2") = py::none(),
             R"pbdoc(
             Load connectivity and delay matrices.

             Parameters
             ----------
             C1 : np.ndarray, shape (N, N), dtype float64
                 Primary normalised weight matrix.
             Delays1 : np.ndarray, shape (N, N), dtype int32
                 Delay indices for C1 (in time-steps).
             C2 : np.ndarray or None
                 Optional secondary weight matrix.
             Delays2 : np.ndarray or None
                 Delay indices for C2. Required when C2 is given.
             )pbdoc")

        .def("simulate",
             &StuartLandauSimulator::simulate,
             py::arg("K"),
             py::arg("a"),
             py::arg("f"),
             R"pbdoc(
             Run the simulation and return the real-part trajectory.

             Parameters
             ----------
             K : float
                 Global coupling strength.
             a : float
                 Bifurcation parameter (negative = damped, positive = limit cycle).
             f : np.ndarray, shape (N,), dtype float64
                 Natural oscillation frequencies (Hz) for each node.

             Returns
             -------
             trajectory : np.ndarray, shape (N, n_save), dtype float64
                 Real part of the complex oscillator state recorded every dt_save.
             )pbdoc");
}
