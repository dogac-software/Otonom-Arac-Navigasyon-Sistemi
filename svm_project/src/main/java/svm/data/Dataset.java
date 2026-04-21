package svm.data;

import svm.model.DataPoint;
import java.util.*;

/**
 * Container for the training dataset.
 * Provides factory methods for built-in scenarios and CSV loading.
 *
 * Time complexity of all factory methods: O(n)
 * Space complexity: O(n)
 */
public final class Dataset {

    private final List<DataPoint> points;
    private final String name;

    public Dataset(String name, List<DataPoint> points) {
        if (points == null || points.isEmpty()) {
            throw new IllegalArgumentException("Dataset must not be empty");
        }
        this.name = Objects.requireNonNull(name, "name");
        this.points = Collections.unmodifiableList(new ArrayList<>(points));
    }

    // -------------------------------------------------------------------------
    // Factory methods
    // -------------------------------------------------------------------------

    /**
     * Linearly separable dataset: two clearly separated clusters.
     * Suitable for demonstrating a wide-margin SVM boundary.
     */
    public static Dataset createLinearlySeperableDemo() {
        List<DataPoint> pts = new ArrayList<>();

        // Class +1 — upper-right cluster (obstacle type A)
        double[][] pos = {
            {2.0, 3.0}, {2.5, 3.5}, {3.0, 3.0}, {3.5, 4.0},
            {2.0, 4.0}, {4.0, 3.5}, {3.0, 4.5}, {2.5, 2.5},
            {4.5, 4.0}, {3.5, 2.5}
        };

        // Class -1 — lower-left cluster (obstacle type B)
        double[][] neg = {
            {-2.0, -3.0}, {-2.5, -3.5}, {-3.0, -3.0}, {-3.5, -4.0},
            {-2.0, -4.0}, {-4.0, -3.5}, {-3.0, -4.5}, {-2.5, -2.5},
            {-4.5, -4.0}, {-3.5, -2.5}
        };

        for (double[] p : pos) pts.add(new DataPoint(p[0], p[1],  1));
        for (double[] p : neg) pts.add(new DataPoint(p[0], p[1], -1));

        return new Dataset("LinearlySeperableDemo", pts);
    }

    /**
     * Narrow-margin scenario: classes are close to the boundary.
     */
    public static Dataset createNarrowMarginDemo() {
        List<DataPoint> pts = new ArrayList<>();

        double[][] pos = {
            {1.0, 0.5}, {1.5, 1.0}, {2.0, 0.0}, {1.0, 1.5}, {2.5, 0.5}
        };
        double[][] neg = {
            {-1.0, 0.5}, {-1.5, 1.0}, {-2.0, 0.0}, {-1.0, 1.5}, {-2.5, 0.5}
        };

        for (double[] p : pos) pts.add(new DataPoint(p[0], p[1],  1));
        for (double[] p : neg) pts.add(new DataPoint(p[0], p[1], -1));

        return new Dataset("NarrowMarginDemo", pts);
    }

    /**
     * Custom dataset from raw double arrays.
     *
     * @param name    dataset name
     * @param coords  Nx2 array of [x, y] coordinates
     * @param labels  N-length array of +1/-1 labels
     */
    public static Dataset fromArrays(String name, double[][] coords, int[] labels) {
        if (coords.length != labels.length) {
            throw new IllegalArgumentException("coords and labels length mismatch");
        }
        List<DataPoint> pts = new ArrayList<>(coords.length);
        for (int i = 0; i < coords.length; i++) {
            pts.add(new DataPoint(coords[i][0], coords[i][1], labels[i]));
        }
        return new Dataset(name, pts);
    }

    /**
     * Randomly generates a linearly separable dataset.
     *
     * @param n      number of points per class
     * @param seed   random seed for reproducibility
     */
    public static Dataset generateRandom(int n, long seed) {
        Random rng = new Random(seed);
        List<DataPoint> pts = new ArrayList<>(2 * n);

        // Positive class: centered at (3, 3)
        for (int i = 0; i < n; i++) {
            double x = 3.0 + rng.nextGaussian() * 0.8;
            double y = 3.0 + rng.nextGaussian() * 0.8;
            pts.add(new DataPoint(x, y, 1));
        }

        // Negative class: centered at (-3, -3)
        for (int i = 0; i < n; i++) {
            double x = -3.0 + rng.nextGaussian() * 0.8;
            double y = -3.0 + rng.nextGaussian() * 0.8;
            pts.add(new DataPoint(x, y, -1));
        }

        return new Dataset("Random(n=" + n + ",seed=" + seed + ")", pts);
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    public List<DataPoint> getPoints()  { return points; }
    public String getName()             { return name; }
    public int size()                   { return points.size(); }

    /** Returns only points with label == +1. O(n) */
    public List<DataPoint> getPositivePoints() {
        List<DataPoint> result = new ArrayList<>();
        for (DataPoint p : points) if (p.getLabel() == 1)  result.add(p);
        return result;
    }

    /** Returns only points with label == -1. O(n) */
    public List<DataPoint> getNegativePoints() {
        List<DataPoint> result = new ArrayList<>();
        for (DataPoint p : points) if (p.getLabel() == -1) result.add(p);
        return result;
    }

    @Override
    public String toString() {
        return String.format("Dataset{name='%s', size=%d}", name, points.size());
    }
}
