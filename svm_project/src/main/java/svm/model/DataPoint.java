package svm.model;

/**
 * Represents a 2D data point with a class label.
 * Used to represent obstacle coordinates in the autonomous vehicle navigation system.
 *
 * Memory: Immutable fields, no leaks possible.
 */
public final class DataPoint {

    private final double x;
    private final double y;
    private final int label; // +1 or -1 (two obstacle classes)

    public DataPoint(double x, double y, int label) {
        if (label != 1 && label != -1) {
            throw new IllegalArgumentException("Label must be +1 or -1, got: " + label);
        }
        this.x = x;
        this.y = y;
        this.label = label;
    }

    public double getX() { return x; }
    public double getY() { return y; }
    public int getLabel() { return label; }

    /** Returns the feature vector as double array [x, y]. */
    public double[] getFeatureVector() {
        return new double[]{x, y};
    }

    @Override
    public String toString() {
        return String.format("DataPoint(x=%.4f, y=%.4f, label=%+d)", x, y, label);
    }
}
