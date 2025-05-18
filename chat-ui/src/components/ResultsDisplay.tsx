import { useMemo } from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

/**
 * ResultsDisplayProps:
 * - results: can be an array of objects, an object with `rows` and `columns`,
 *   or an object of arrays
 */
interface ResultsDisplayProps {
  results: any;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ results }) => {
  // Normalize results into array of row objects
  const rows = useMemo<any[]>(() => {
    // Case 1: results is already an array of objects
    if (Array.isArray(results)) {
      return results;
    }
    // Case 2: results has .rows (array) and optionally .columns
    if (results.rows && Array.isArray(results.rows)) {
      return results.rows;
    }
    // Case 3: object of arrays { col1: [...], col2: [...] }
    const cols: string[] = Object.keys(results).filter((key: string) => Array.isArray(results[key]));
    const length: number = results[cols[0]]?.length || 0;
    const arr: any[] = [];
    for (let i = 0; i < length; i++) {
      const row: any = {};
      cols.forEach((col: string) => {
        row[col] = results[col][i];
      });
      arr.push(row);
    }
    return arr;
  }, [results]);

  if (!rows || rows.length === 0) return null;

  // Determine columns: if provided, use results.columns; else keys of first row
  const columns = useMemo<string[]>(() => {
    if (results.columns && Array.isArray(results.columns)) {
      return results.columns;
    }
    return Object.keys(rows[0]);
  }, [results, rows]);

  const numRows: number = rows.length;

  // Simple chart heuristic: 2–5 columns, <=20 rows, first col strings, second numeric
  const canVisualize = useMemo<boolean>(() => {
    if (columns.length >= 2 && columns.length <= 5 && numRows > 0 && numRows <= 20) {
      const firstVals = rows.map((r: any) => r[columns[0]]);
      const secondVals = rows.map((r: any) => r[columns[1]]);
      const firstIsStr = firstVals.every((v: any) => typeof v === 'string');
      const secondIsNum = secondVals.every((v: any) => typeof v === 'number' || !isNaN(Number(v)));
      return firstIsStr && secondIsNum;
    }
    return false;
  }, [columns, numRows, rows]);

  // Build chart data if possible
  const chartData = useMemo<any | null>(() => {
    if (!canVisualize) return null;
    return {
      labels: rows.map((r: any) => r[columns[0]]),
      datasets: columns.slice(1).map((col: string, idx: number) => ({
        label: col,
        data: rows.map((r: any) => Number(r[col] ?? 0)),
        backgroundColor: `rgba(${(idx * 50) % 255}, ${(idx * 80) % 255}, ${(idx * 110) % 255}, 0.7)`,
        borderColor: `rgba(${(idx * 50) % 255}, ${(idx * 80) % 255}, ${(idx * 110) % 255}, 1)`,
        borderWidth: 1,
      })),
    };
  }, [canVisualize, columns, rows]);

  return (
    <div className="results-container">
      {canVisualize && chartData && (
        <div className="chart-container">
          <Bar
            data={chartData}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Query Results' },
              },
            }}
          />
        </div>
      )}

      <div className="table-container">
        <table className="result-table">
          <thead>
            <tr>
              {columns.map((col: string) => (
                <th key={col}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row: any, idx: number) => (
              <tr key={idx}>
                {columns.map((col: string) => (
                  <td key={`${col}-${idx}`}>{
                    // If nested object, stringify
                    typeof row[col] === 'object' && row[col] !== null
                      ? JSON.stringify(row[col])
                      : row[col]?.toString()
                  }</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ResultsDisplay;
