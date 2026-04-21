package svm.visualization;

import svm.data.Dataset;
import svm.model.DataPoint;
import svm.model.SVMResult;

import java.util.List;

/**
 * Renders an ASCII scatter plot of the dataset and the SVM decision boundary.
 *
 * Legend:
 *   +  positive class (+1)
 *   -  negative class (-1)
 *   S  support vector
 *   |  decision boundary crossing
 *   .  empty cell
 *
 * Time complexity: O(W × H + n)  where W, H are grid dimensions and n is dataset size.
 */
public final class ConsoleVisualizer {

    private final int width;
    private final int height;

    public ConsoleVisualizer() {
        this(60, 25);
    }

    public ConsoleVisualizer(int width, int height) {
        if (width < 10 || height < 5) throw new IllegalArgumentException("Grid too small");
        this.width  = width;
        this.height = height;
    }

    /**
     * Renders the dataset and SVM result to a String.
     */
    public String render(Dataset dataset, SVMResult result) {
        List<DataPoint> pts = dataset.getPoints();

        // Compute bounding box
        double xMin = Double.MAX_VALUE, xMax = -Double.MAX_VALUE;
        double yMin = Double.MAX_VALUE, yMax = -Double.MAX_VALUE;
        for (DataPoint p : pts) {
            xMin = Math.min(xMin, p.getX()); xMax = Math.max(xMax, p.getX());
            yMin = Math.min(yMin, p.getY()); yMax = Math.max(yMax, p.getY());
        }

        // Extend bounds to show margin lines
        double padX = (xMax - xMin) * 0.2 + 1;
        double padY = (yMax - yMin) * 0.2 + 1;
        xMin -= padX; xMax += padX;
        yMin -= padY; yMax += padY;

        // Build empty grid
        char[][] grid = new char[height][width];
        for (int r = 0; r < height; r++)
            for (int c = 0; c < width; c++)
                grid[r][c] = ' ';

        // Draw axes
        int axisRow = worldToRow(0, yMin, yMax);
        int axisCol = worldToCol(0, xMin, xMax);
        if (axisRow >= 0 && axisRow < height)
            for (int c = 0; c < width; c++) grid[axisRow][c] = '─';
        if (axisCol >= 0 && axisCol < width)
            for (int r = 0; r < height; r++) grid[r][axisCol] = '│';
        if (axisRow >= 0 && axisRow < height && axisCol >= 0 && axisCol < width)
            grid[axisRow][axisCol] = '┼';

        // Draw decision boundary and margin lines
        double[] w = result.getWeights();
        double   b = result.getBias();
        double normW = Math.sqrt(w[0]*w[0] + w[1]*w[1]);

        for (int c = 0; c < width; c++) {
            double x = colToWorld(c, xMin, xMax);
            if (Math.abs(w[1]) > 1e-10) {
                // Decision boundary: w[0]*x + w[1]*y + b = 0 => y = -(w[0]*x + b) / w[1]
                double yBound  = -(w[0]*x + b)          / w[1];
                double yMarP   = -(w[0]*x + b - 1.0)    / w[1];
                double yMarN   = -(w[0]*x + b + 1.0)    / w[1];

                plotRow(grid, yBound,  yMin, yMax, c, '|');
                plotRow(grid, yMarP,   yMin, yMax, c, ':');
                plotRow(grid, yMarN,   yMin, yMax, c, ':');
            }
        }

        // Mark support vectors first (capital S)
        for (DataPoint sv : result.getSupportVectors()) {
            int r = worldToRow(sv.getY(), yMin, yMax);
            int c = worldToCol(sv.getX(), xMin, xMax);
            if (inBounds(r, c)) grid[r][c] = 'S';
        }

        // Plot data points
        for (DataPoint p : pts) {
            int r = worldToRow(p.getY(), yMin, yMax);
            int c = worldToCol(p.getX(), xMin, xMax);
            if (!inBounds(r, c)) continue;
            if (grid[r][c] == 'S') continue; // support vector already marked
            grid[r][c] = (p.getLabel() == 1) ? '+' : '-';
        }

        // Render to string
        StringBuilder sb = new StringBuilder();
        sb.append("┌").append("─".repeat(width)).append("┐\n");
        for (int r = 0; r < height; r++) {
            sb.append('│');
            sb.append(new String(grid[r]));
            sb.append("│\n");
        }
        sb.append("└").append("─".repeat(width)).append("┘\n");
        sb.append(String.format("  x: [%.2f, %.2f]   y: [%.2f, %.2f]%n", xMin, xMax, yMin, yMax));
        sb.append("  Legend: + pos class   - neg class   S support vector\n");
        sb.append("          | decision boundary   : margin lines\n");
        return sb.toString();
    }

    // ── Coordinate helpers ────────────────────────────────────────────────────

    private int worldToRow(double y, double yMin, double yMax) {
        // Screen row 0 = top = yMax
        return (int) Math.round((yMax - y) / (yMax - yMin) * (height - 1));
    }

    private int worldToCol(double x, double xMin, double xMax) {
        return (int) Math.round((x - xMin) / (xMax - xMin) * (width - 1));
    }

    private double colToWorld(int c, double xMin, double xMax) {
        return xMin + (double) c / (width - 1) * (xMax - xMin);
    }

    private void plotRow(char[][] grid, double y, double yMin, double yMax, int col, char ch) {
        int r = worldToRow(y, yMin, yMax);
        if (inBounds(r, col) && grid[r][col] == ' ') grid[r][col] = ch;
    }

    private boolean inBounds(int r, int c) {
        return r >= 0 && r < height && c >= 0 && c < width;
    }
}
