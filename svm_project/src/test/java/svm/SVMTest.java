package svm;

import svm.algorithm.SVMAlgorithm;
import svm.algorithm.SVMValidator;
import svm.data.Dataset;
import svm.model.DataPoint;
import svm.model.SVMResult;

/**
 * Lightweight unit tests (no external test framework required).
 * Run: java -cp . svm.SVMTest
 *
 * Tests:
 *   T1 — Wide-separation: accuracy = 100%, margin > 0
 *   T2 — Narrow-margin:   accuracy = 100%, constraints satisfied
 *   T3 — Random dataset:  accuracy = 100%
 *   T4 — Predict() works correctly for known points
 *   T5 — DataPoint label validation (IllegalArgumentException for invalid label)
 *   T6 — Custom coordinate dataset
 */
public class SVMTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("═══════════════════════════════════════");
        System.out.println("  SVM Unit Tests");
        System.out.println("═══════════════════════════════════════");

        test1_wideSeparation();
        test2_narrowMargin();
        test3_randomDataset();
        test4_prediction();
        test5_invalidLabel();
        test6_customCoordinates();

        System.out.println("───────────────────────────────────────");
        System.out.printf("  Passed: %d  |  Failed: %d%n", passed, failed);
        System.out.println("═══════════════════════════════════════");
        if (failed > 0) System.exit(1);
    }

    // ── Tests ────────────────────────────────────────────────────────────────

    static void test1_wideSeparation() {
        Dataset      ds     = Dataset.createLinearlySeperableDemo();
        SVMAlgorithm svm    = new SVMAlgorithm();
        SVMResult    result = svm.fit(ds);

        assertTrue("T1a: accuracy=100%",  SVMValidator.accuracy(result, ds) == 1.0);
        assertTrue("T1b: margin > 0",     result.getMargin() > 0);
        assertTrue("T1c: constraints ok", SVMValidator.allConstraintsSatisfied(result, ds));
        assertTrue("T1d: converged",      result.isConverged());
    }

    static void test2_narrowMargin() {
        Dataset      ds     = Dataset.createNarrowMarginDemo();
        SVMAlgorithm svm    = new SVMAlgorithm();
        SVMResult    result = svm.fit(ds);

        assertTrue("T2a: accuracy=100%",  SVMValidator.accuracy(result, ds) == 1.0);
        assertTrue("T2b: margin > 0",     result.getMargin() > 0);
        assertTrue("T2c: constraints ok", SVMValidator.allConstraintsSatisfied(result, ds));
    }

    static void test3_randomDataset() {
        Dataset      ds     = Dataset.generateRandom(20, 7L);
        SVMAlgorithm svm    = new SVMAlgorithm();
        SVMResult    result = svm.fit(ds);

        assertTrue("T3a: accuracy=100%",  SVMValidator.accuracy(result, ds) == 1.0);
        assertTrue("T3b: margin > 0",     result.getMargin() > 0);
    }

    static void test4_prediction() {
        // Trivial perfectly separable: class +1 at x>0, class -1 at x<0
        double[][] coords = {{3,0},{3,1},{4,0},{4,1},{-3,0},{-3,1},{-4,0},{-4,1}};
        int[]      labels = {1,1,1,1,-1,-1,-1,-1};
        Dataset      ds     = Dataset.fromArrays("T4", coords, labels);
        SVMAlgorithm svm    = new SVMAlgorithm();
        SVMResult    result = svm.fit(ds);

        // Predict known points
        DataPoint posPoint = new DataPoint(10, 0, 1);
        DataPoint negPoint = new DataPoint(-10, 0, -1);
        assertTrue("T4a: predict positive",  result.predict(posPoint) ==  1);
        assertTrue("T4b: predict negative",  result.predict(negPoint) == -1);
    }

    static void test5_invalidLabel() {
        boolean threw = false;
        try {
            new DataPoint(0, 0, 0); // label 0 is invalid
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        assertTrue("T5: IllegalArgumentException for label=0", threw);
    }

    static void test6_customCoordinates() {
        double[][] coords = {
            { 1, 4},{ 2, 3},{ 3, 5},{-1,-4},{-2,-3},{-3,-5}
        };
        int[] labels = {1,1,1,-1,-1,-1};
        Dataset      ds     = Dataset.fromArrays("T6", coords, labels);
        SVMAlgorithm svm    = new SVMAlgorithm();
        SVMResult    result = svm.fit(ds);

        assertTrue("T6a: accuracy=100%", SVMValidator.accuracy(result, ds) == 1.0);
        assertTrue("T6b: margin > 0",    result.getMargin() > 0);
    }

    // ── Assertion helpers ────────────────────────────────────────────────────

    static void assertTrue(String name, boolean condition) {
        if (condition) {
            System.out.printf("  [PASS] %s%n", name);
            passed++;
        } else {
            System.out.printf("  [FAIL] %s%n", name);
            failed++;
        }
    }
}
