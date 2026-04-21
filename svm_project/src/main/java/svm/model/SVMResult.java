package svm.model;

import java.util.Collections;
import java.util.List;

/**
 * Immutable result object produced by the SVM algorithm.
 * Contains the hyperplane parameters (w, b), margin width, and support vectors.
 *
 * Hyperplane equation: w[0]*x + w[1]*y + b = 0
 * Decision boundary:   sign(w·x + b)
 */
public final class SVMResult {

    private final double[] weights;      // w vector [w0, w1]
    private final double bias;           // b (intercept)
    private final double margin;         // 2 / ||w||  (total margin width)
    private final List<DataPoint> supportVectors;
    private final int iterations;
    private final boolean converged;

    public SVMResult(double[] weights, double bias, double margin,
                     List<DataPoint> supportVectors, int iterations, boolean converged) {
        if (weights == null || weights.length != 2) {
            throw new IllegalArgumentException("weights must be a 2-element array");
        }
        this.weights = weights.clone(); // defensive copy
        this.bias = bias;
        this.margin = margin;
        this.supportVectors = Collections.unmodifiableList(supportVectors);
        this.iterations = iterations;
        this.converged = converged;
    }

    /** Predicts the class label for a given point. Returns +1 or -1. */
    public int predict(DataPoint p) {
        double score = weights[0] * p.getX() + weights[1] * p.getY() + bias;
        return score >= 0 ? 1 : -1;
    }

    /** Returns the raw decision function value (signed distance * ||w||). */
    public double decisionFunction(DataPoint p) {
        return weights[0] * p.getX() + weights[1] * p.getY() + bias;
    }

    /** Geometric distance from a point to the decision hyperplane. */
    public double distanceToHyperplane(DataPoint p) {
        double norm = Math.sqrt(weights[0] * weights[0] + weights[1] * weights[1]);
        if (norm < 1e-12) return Double.POSITIVE_INFINITY;
        return Math.abs(decisionFunction(p)) / norm;
    }

    public double[] getWeights()              { return weights.clone(); }
    public double getBias()                   { return bias; }
    public double getMargin()                 { return margin; }
    public List<DataPoint> getSupportVectors(){ return supportVectors; }
    public int getIterations()                { return iterations; }
    public boolean isConverged()              { return converged; }

    /** y-intercept of the decision boundary line (w[0]*x + w[1]*y + b = 0). */
    public double getDecisionLineYIntercept() {
        if (Math.abs(weights[1]) < 1e-12) return Double.NaN;
        return -bias / weights[1];
    }

    /** Slope of the decision boundary line. */
    public double getDecisionLineSlope() {
        if (Math.abs(weights[1]) < 1e-12) return Double.NaN;
        return -weights[0] / weights[1];
    }

    @Override
    public String toString() {
        return String.format(
            "SVMResult{w=[%.6f, %.6f], b=%.6f, margin=%.6f, supportVectors=%d, iterations=%d, converged=%b}",
            weights[0], weights[1], bias, margin, supportVectors.size(), iterations, converged
        );
    }
}
