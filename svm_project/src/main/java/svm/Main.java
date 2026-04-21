package svm;

import svm.algorithm.SVMAlgorithm;
import svm.algorithm.SVMValidator;
import svm.data.Dataset;
import svm.model.SVMResult;
import svm.utils.ComplexityAnalyzer;
import svm.visualization.ConsoleVisualizer;

/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Autonomous Vehicle Navigation — Maximum-Margin Safety Boundary System
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Entry point demonstrating:
 *   1. Standard linearly-separable scenario
 *   2. Narrow-margin scenario
 *   3. Random dataset
 *   4. Complexity benchmark
 *
 * Run: java -cp . svm.Main
 */
public class Main {

    public static void main(String[] args) {

        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════╗");
        System.out.println("║   Autonomous Vehicle Safety Module — SVM Boundary Finder         ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════╝");
        System.out.println();

        SVMAlgorithm       svm  = new SVMAlgorithm();
        ConsoleVisualizer  viz  = new ConsoleVisualizer();

        // ── Scenario 1: Wide-separation demo ─────────────────────────────────
        runScenario("Scenario 1 — Wide Separation (Demo Dataset)", svm, viz,
                Dataset.createLinearlySeperableDemo());

        // ── Scenario 2: Narrow-margin demo ───────────────────────────────────
        runScenario("Scenario 2 — Narrow Margin", svm, viz,
                Dataset.createNarrowMarginDemo());

        // ── Scenario 3: Random dataset ────────────────────────────────────────
        runScenario("Scenario 3 — Random Dataset (n=30, seed=42)", svm, viz,
                Dataset.generateRandom(15, 42L));

        // ── Scenario 4: Custom coordinates from assignment ────────────────────
        runCustomCoordinates(svm, viz);

        // ── Complexity analysis ───────────────────────────────────────────────
        System.out.println(ComplexityAnalyzer.formalAnalysis());
        System.out.println("Running empirical benchmark (this may take a few seconds)...");
        ComplexityAnalyzer.runBenchmark(new int[]{10, 20, 40, 80}, 99L);
    }

    // =========================================================================

    private static void runScenario(String title,
                                    SVMAlgorithm svm,
                                    ConsoleVisualizer viz,
                                    Dataset dataset) {
        System.out.println("─".repeat(68));
        System.out.println("  " + title);
        System.out.println("─".repeat(68));
        System.out.println("  Dataset: " + dataset);
        System.out.println();

        SVMResult result = svm.fit(dataset);

        System.out.println(viz.render(dataset, result));
        System.out.println(SVMValidator.report(result, dataset));
    }

    private static void runCustomCoordinates(SVMAlgorithm svm, ConsoleVisualizer viz) {
        // Two obstacle classes as 2D coordinates
        double[][] coords = {
            // Class +1 (obstacle type A)
            { 1.0,  4.0}, { 2.0,  3.5}, { 3.0,  5.0}, { 1.5,  5.5}, { 2.5,  4.5},
            // Class -1 (obstacle type B)
            {-1.0, -4.0}, {-2.0, -3.5}, {-3.0, -5.0}, {-1.5, -5.5}, {-2.5, -4.5}
        };
        int[] labels = { 1, 1, 1, 1, 1, -1, -1, -1, -1, -1};

        Dataset dataset = Dataset.fromArrays("CustomCoordinates", coords, labels);
        runScenario("Scenario 4 — Custom Obstacle Coordinates", svm, viz, dataset);
    }
}
