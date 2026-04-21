package svm.algorithm;

import svm.data.Dataset;
import svm.model.DataPoint;
import svm.model.SVMResult;

import java.util.List;

/**
 * Validates an SVMResult against a dataset.
 *
 * Checks:
 *  1. All points are correctly classified.
 *  2. All points satisfy the margin constraint: yᵢ(w·xᵢ + b) ≥ 1 − ε
 *  3. Support vectors lie exactly on the margin: yᵢ(w·xᵢ + b) ≈ 1
 *
 * Time complexity: O(n) per call.
 */
public final class SVMValidator {

    private static final double MARGIN_EPSILON = 1e-3;

    private SVMValidator() {} // utility class – no instantiation

    /**
     * Validates classification accuracy.
     *
     * @return fraction of correctly classified points in [0, 1]
     */
    public static double accuracy(SVMResult result, Dataset dataset) {
        List<DataPoint> pts = dataset.getPoints();
        int correct = 0;
        for (DataPoint p : pts) {
            if (result.predict(p) == p.getLabel()) correct++;
        }
        return (double) correct / pts.size();
    }

    /**
     * Checks that all margin constraints are satisfied.
     *
     * @return true if yᵢ(w·xᵢ + b) ≥ 1 − ε for all points
     */
    public static boolean allConstraintsSatisfied(SVMResult result, Dataset dataset) {
        double[] w = result.getWeights();
        double   b = result.getBias();

        for (DataPoint p : dataset.getPoints()) {
            double functional = p.getLabel() * (w[0] * p.getX() + w[1] * p.getY() + b);
            if (functional < 1.0 - MARGIN_EPSILON) return false;
        }
        return true;
    }

    /**
     * Returns a human-readable validation report.
     */
    public static String report(SVMResult result, Dataset dataset) {
        StringBuilder sb = new StringBuilder();
        sb.append("═══════════════════════════════════════\n");
        sb.append("  SVM Validation Report\n");
        sb.append("═══════════════════════════════════════\n");
        sb.append(String.format("  Dataset       : %s%n", dataset.getName()));
        sb.append(String.format("  Points        : %d%n", dataset.size()));
        sb.append(String.format("  Accuracy      : %.2f%%%n", accuracy(result, dataset) * 100));
        sb.append(String.format("  Constraints   : %s%n",
                allConstraintsSatisfied(result, dataset) ? "ALL SATISFIED ✓" : "VIOLATED ✗"));
        sb.append(String.format("  Converged     : %b%n", result.isConverged()));
        sb.append(String.format("  Iterations    : %d%n", result.getIterations()));
        sb.append(String.format("  Margin Width  : %.6f%n", result.getMargin()));
        sb.append(String.format("  Support Vec.  : %d%n", result.getSupportVectors().size()));

        double[] w = result.getWeights();
        double   b = result.getBias();
        sb.append(String.format("  Hyperplane    : %.4f·x + %.4f·y + (%.4f) = 0%n", w[0], w[1], b));
        sb.append(String.format("  Slope         : %.4f%n", result.getDecisionLineSlope()));
        sb.append(String.format("  Y-intercept   : %.4f%n", result.getDecisionLineYIntercept()));
        sb.append("═══════════════════════════════════════\n");
        return sb.toString();
    }
}
