package svm.utils;

import svm.algorithm.SVMAlgorithm;
import svm.data.Dataset;
import svm.model.SVMResult;

/**
 * Empirically measures training time vs. dataset size to validate the
 * theoretical O(n²) time complexity of the SVM algorithm.
 *
 * Also contains the formal Big-O analysis as documentation.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * FORMAL COMPLEXITY ANALYSIS
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Let n = number of training points, d = feature dimensions (d = 2 here).
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │ Operation                   │ Time          │ Space             │
 * ├─────────────────────────────────────────────────────────────────┤
 * │ initState()                 │ O(n)          │ O(n·d) = O(n)     │
 * │ kernel(i, j)                │ O(d) = O(1)   │ O(1)              │
 * │ decisionValue(i)            │ O(n)          │ O(1)              │
 * │ selectSecondAlpha(i, Ei)    │ O(n)          │ O(1)              │
 * │ Single SMO outer iteration  │ O(n²)         │ O(1) extra        │
 * │   (n inner × O(n) decision) │               │                   │
 * │ Total iterations (worst)    │ O(n) passes   │ —                 │
 * │ Total SMO training          │ O(n³) worst   │ O(n)              │
 * │ Total SMO training (avg)    │ O(n²)         │ O(n)              │
 * │ buildResult()               │ O(n)          │ O(s) s=|SVs|      │
 * ├─────────────────────────────────────────────────────────────────┤
 * │ OVERALL                     │ O(n²) avg     │ O(n)              │
 * │                             │ O(n³) worst   │                   │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * Why O(n²) average?
 *   For linearly separable data with a clear margin, SMO converges in O(n)
 *   passes. Each pass scans all n points and each KKT check / alpha update
 *   costs O(n) due to the decision function evaluation. Hence O(n) × O(n) = O(n²).
 *
 * Why is this optimal?
 *   The SVM margin maximisation objective directly minimises ||w||², which
 *   uniquely maximises the geometric margin 2/||w||. The resulting hyperplane
 *   is equidistant from the nearest point in each class (the support vectors).
 *   No other separating hyperplane has a larger margin — this is proven by the
 *   convexity of the optimisation problem (QP with a unique global optimum).
 * ─────────────────────────────────────────────────────────────────────────────
 */
public final class ComplexityAnalyzer {

    private ComplexityAnalyzer() {}

    /**
     * Runs the SVM for increasing dataset sizes and prints timing results.
     *
     * @param sizes   array of n values to benchmark
     * @param seed    random seed for reproducibility
     */
    public static void runBenchmark(int[] sizes, long seed) {
        System.out.println("════════════════════════════════════════════════════════════════");
        System.out.println("  Time Complexity Benchmark  (expected: O(n²) average)");
        System.out.println("════════════════════════════════════════════════════════════════");
        System.out.printf("  %-8s  %-12s  %-12s  %-10s%n", "n", "Time (ms)", "Ratio", "Expected");
        System.out.println("  ─────────────────────────────────────────────────────────────");

        long prevTime = -1;
        int  prevN    = -1;

        for (int n : sizes) {
            Dataset  ds  = Dataset.generateRandom(n, seed);
            SVMAlgorithm svm = new SVMAlgorithm(50_000, 1e-4, 1e-5);

            long start  = System.nanoTime();
            svm.fit(ds);
            long elapsed = (System.nanoTime() - start) / 1_000_000; // ms

            String ratio    = "—";
            String expected = "—";
            if (prevTime > 0 && elapsed > 0) {
                double r = (double) elapsed / prevTime;
                double e = (double) n / prevN;
                ratio    = String.format("%.2f×", r);
                expected = String.format("%.2f×", e * e); // O(n²) → (n/prevN)²
            }

            System.out.printf("  %-8d  %-12d  %-12s  %-10s%n", n, elapsed, ratio, expected);
            prevTime = elapsed;
            prevN    = n;
        }

        System.out.println("════════════════════════════════════════════════════════════════");
        System.out.println("  Ratio ≈ Expected confirms O(n²) empirically.");
        System.out.println("════════════════════════════════════════════════════════════════");
    }

    /** Returns the formal Big-O complexity as a formatted string. */
    public static String formalAnalysis() {
        return """
                ╔══════════════════════════════════════════════════════════════╗
                ║          FORMAL BIG-O ANALYSIS — Hard-Margin SVM (SMO)      ║
                ╠══════════════════════════════════════════════════════════════╣
                ║  Training (average, well-separated data)  :  O(n²)          ║
                ║  Training (worst case)                    :  O(n³)          ║
                ║  Space complexity                         :  O(n)           ║
                ║  Prediction per point                     :  O(s)  s=|SVs|  ║
                ╠══════════════════════════════════════════════════════════════╣
                ║  Justification:                                              ║
                ║  • Each full pass over n points costs O(n) decision evals   ║
                ║  • Each decision eval sums over n support vectors: O(n)     ║
                ║  • For separable data: O(n) passes until convergence        ║
                ║  • Total: O(n) passes × O(n²) per pass = O(n³) worst case  ║
                ║           O(1) passes typically → O(n²) average             ║
                ╚══════════════════════════════════════════════════════════════╝
                """;
    }
}
