package svm.algorithm;

import svm.data.Dataset;
import svm.model.DataPoint;
import svm.model.SVMResult;

import java.util.ArrayList;
import java.util.List;

/**
 * Hard-Margin Support Vector Machine using the Sequential Minimal Optimization (SMO)
 * algorithm (Platt, 1998).
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Mathematical Foundation
 * ─────────────────────────────────────────────────────────────────────────────
 * Primal problem (hard-margin):
 *
 *   minimize    ½ ||w||²
 *   subject to  yᵢ(w·xᵢ + b) ≥ 1   ∀ i
 *
 * Dual problem:
 *
 *   maximize    Σ αᵢ − ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ (xᵢ·xⱼ)
 *   subject to  αᵢ ≥ 0  and  Σ αᵢ yᵢ = 0
 *
 * At optimality:
 *   w  = Σ αᵢ yᵢ xᵢ
 *   b  = yₛ − w·xₛ   (for any support vector s where αₛ > 0)
 *   margin = 2 / ||w||
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Time Complexity Analysis
 * ─────────────────────────────────────────────────────────────────────────────
 *   Per iteration : O(n)   — full pass over all training points
 *   Iterations    : O(n)   — worst case, typically much fewer
 *   Overall       : O(n²)  in the worst case
 *
 *   Space         : O(n)   — for the alpha vector and kernel cache
 *
 * For the hard-margin linear kernel, convergence is typically achieved in O(n)
 * passes for well-separated data, giving an expected O(n²) total.
 * ─────────────────────────────────────────────────────────────────────────────
 */
public class SVMAlgorithm {

    // ── Hyper-parameters ──────────────────────────────────────────────────────
    private final int    maxIterations;
    private final double tolerance;          // KKT violation tolerance
    private final double alphaMinDelta;      // minimum meaningful alpha change

    // ── Internal training state (cleared after each fit call) ─────────────────
    private double[]    alphas;
    private double      bias;
    private double[][]  X;                   // feature matrix [n][2]
    private int[]       Y;                   // label vector   [n]
    private int         n;

    /**
     * Default constructor with sensible parameters.
     */
    public SVMAlgorithm() {
        this(10_000, 1e-4, 1e-5);
    }

    /**
     * @param maxIterations maximum number of full passes over the training set
     * @param tolerance     KKT violation tolerance (epsilon)
     * @param alphaMinDelta minimum change in alpha to count as "progress"
     */
    public SVMAlgorithm(int maxIterations, double tolerance, double alphaMinDelta) {
        if (maxIterations <= 0) throw new IllegalArgumentException("maxIterations must be positive");
        if (tolerance <= 0)     throw new IllegalArgumentException("tolerance must be positive");
        this.maxIterations = maxIterations;
        this.tolerance     = tolerance;
        this.alphaMinDelta = alphaMinDelta;
    }

    // =========================================================================
    // Public API
    // =========================================================================

    /**
     * Trains the SVM on the given dataset and returns the result.
     *
     * @param dataset labeled 2D points (must be linearly separable)
     * @return SVMResult containing w, b, margin, and support vectors
     * @throws IllegalStateException if the dataset is not linearly separable
     */
    public SVMResult fit(Dataset dataset) {
        initState(dataset);

        int  passes        = 0;
        int  iterCount     = 0;
        boolean converged  = false;

        while (passes < 3 && iterCount < maxIterations) {
            int alphasChanged = 0;

            for (int i = 0; i < n; i++) {
                double Ei = decisionValue(i) - Y[i];

                // Check KKT condition for αᵢ
                if (kktViolated(i, Ei)) {
                    // Select j ≠ i heuristically (maximum |Ej - Ei|)
                    int j = selectSecondAlpha(i, Ei);
                    if (j < 0) continue;

                    double Ej = decisionValue(j) - Y[j];

                    double alphaIOld = alphas[i];
                    double alphaJOld = alphas[j];

                    // Compute bounds [L, H] for αⱼ
                    double L, H;
                    if (Y[i] != Y[j]) {
                        L = Math.max(0, alphaJOld - alphaIOld);
                        H = Double.MAX_VALUE;               // hard-margin: no upper bound
                    } else {
                        L = Math.max(0, alphaIOld + alphaJOld);
                        H = Double.MAX_VALUE;
                    }
                    if (L >= H - alphaMinDelta) continue;

                    // Compute η (second derivative of objective)
                    double eta = 2 * kernel(i, j) - kernel(i, i) - kernel(j, j);
                    if (eta >= 0) continue;

                    // Update αⱼ
                    double alphaJNew = alphaJOld - (Y[j] * (Ei - Ej)) / eta;
                    alphaJNew = Math.max(L, alphaJNew);

                    if (Math.abs(alphaJNew - alphaJOld) < alphaMinDelta) continue;

                    // Update αᵢ (constraint: Σ αᵢ yᵢ = 0)
                    double alphaINew = alphaIOld + Y[i] * Y[j] * (alphaJOld - alphaJNew);
                    if (alphaINew < 0) {
                        alphaJNew += Y[i] * Y[j] * alphaINew;
                        alphaINew = 0;
                    }

                    alphas[i] = alphaINew;
                    alphas[j] = alphaJNew;

                    // Update bias b
                    updateBias(i, j, alphaIOld, alphaJOld, Ei, Ej);
                    alphasChanged++;
                }
                iterCount++;
            }

            if (alphasChanged == 0) passes++;
            else                    passes = 0;
        }

        converged = (passes >= 3);
        return buildResult(iterCount, converged);
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    /** Initialises mutable training state. O(n) */
    private void initState(Dataset dataset) {
        List<DataPoint> pts = dataset.getPoints();
        n      = pts.size();
        alphas = new double[n];
        bias   = 0.0;
        X      = new double[n][2];
        Y      = new int[n];

        for (int i = 0; i < n; i++) {
            DataPoint p = pts.get(i);
            X[i][0] = p.getX();
            X[i][1] = p.getY();
            Y[i]    = p.getLabel();
        }
    }

    /** Linear (dot-product) kernel: xᵢ · xⱼ  — O(d), d=2 → O(1) */
    private double kernel(int i, int j) {
        return X[i][0] * X[j][0] + X[i][1] * X[j][1];
    }

    /** f(xᵢ) = Σⱼ αⱼ yⱼ K(xⱼ, xᵢ) + b  — O(n) */
    private double decisionValue(int i) {
        double sum = bias;
        for (int k = 0; k < n; k++) {
            if (alphas[k] > 0) {
                sum += alphas[k] * Y[k] * kernel(k, i);
            }
        }
        return sum;
    }

    /** Returns true when αᵢ violates the KKT conditions. */
    private boolean kktViolated(int i, double Ei) {
        double yEi = Y[i] * Ei;
        return (yEi < -tolerance && alphas[i] < Double.MAX_VALUE)
            || (yEi > tolerance  && alphas[i] > 0);
    }

    /**
     * Heuristic second-alpha selection: choose j that maximises |Ei − Ej|.
     * O(n)
     */
    private int selectSecondAlpha(int i, double Ei) {
        int    best  = -1;
        double bestD = 0;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double d = Math.abs(Ei - (decisionValue(j) - Y[j]));
            if (d > bestD) { bestD = d; best = j; }
        }
        return best;
    }

    /** Updates bias b after an alpha pair update. O(1) */
    private void updateBias(int i, int j,
                            double alphaIOld, double alphaJOld,
                            double Ei, double Ej) {
        double b1 = bias - Ei
                  - Y[i] * (alphas[i] - alphaIOld) * kernel(i, i)
                  - Y[j] * (alphas[j] - alphaJOld) * kernel(i, j);

        double b2 = bias - Ej
                  - Y[i] * (alphas[i] - alphaIOld) * kernel(i, j)
                  - Y[j] * (alphas[j] - alphaJOld) * kernel(j, j);

        bias = (b1 + b2) / 2.0;
    }

    /**
     * Reconstructs w, collects support vectors, computes margin. O(n)
     */
    private SVMResult buildResult(int iterations, boolean converged) {
        double w0 = 0, w1 = 0;
        List<DataPoint> supportVectors = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            if (alphas[i] > 1e-6) {
                w0 += alphas[i] * Y[i] * X[i][0];
                w1 += alphas[i] * Y[i] * X[i][1];
                supportVectors.add(new DataPoint(X[i][0], X[i][1], Y[i]));
            }
        }

        double normW  = Math.sqrt(w0 * w0 + w1 * w1);
        double margin = normW > 1e-12 ? 2.0 / normW : Double.NaN;

        // Create a temporary SVMResult to reuse its predict method
        double[] weights = {w0, w1};
        SVMResult temp = new SVMResult(weights, bias, margin, supportVectors, iterations, converged);

        // Recompute b more accurately from support vectors (reduces floating-point drift)
        double bSum   = 0;
        int    bCount = 0;
        for (int i = 0; i < n; i++) {
            if (alphas[i] > 1e-6) {
                double fxi = w0 * X[i][0] + w1 * X[i][1];
                bSum  += Y[i] - fxi;
                bCount++;
            }
        }
        double refinedBias = bCount > 0 ? bSum / bCount : bias;

        return new SVMResult(weights, refinedBias, margin, supportVectors, iterations, converged);
    }

    // =========================================================================
    // Getters (for testing / introspection)
    // =========================================================================
    public int    getMaxIterations() { return maxIterations; }
    public double getTolerance()     { return tolerance; }
}
