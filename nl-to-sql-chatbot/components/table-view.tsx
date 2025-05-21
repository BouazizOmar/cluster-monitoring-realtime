import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

interface TableViewProps {
  results: any
}

export function TableView({ results }: TableViewProps) {
  if (!results) {
    return <p className="text-blue-800">No results available</p>
  }

  try {
    // Parse results if it's a string
    const parsedResults = typeof results === "string" ? JSON.parse(results) : results

    // Check if we have the expected structure
    if (!parsedResults.rows || !Array.isArray(parsedResults.rows)) {
      return (
        <div className="overflow-x-auto bg-blue-50 p-3 rounded-md border border-blue-100">
          <pre className="text-xs text-blue-800">{JSON.stringify(results, null, 2)}</pre>
        </div>
      )
    }

    return (
      <div>
        <div className="rounded-md border border-blue-100 overflow-hidden">
          <Table>
            <TableHeader className="bg-blue-50">
              <TableRow>
                {parsedResults.columns.map((column: string) => (
                  <TableHead key={column} className="text-blue-700 font-medium">
                    {column}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {parsedResults.rows.map((row: any, index: number) => (
                <TableRow key={index} className="hover:bg-blue-50">
                  {parsedResults.columns.map((column: string) => (
                    <TableCell key={column} className="text-blue-800">
                      {row[column]}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        <div className="mt-2 text-sm text-blue-600">
          Total rows: {parsedResults.row_count || parsedResults.rows.length}
        </div>
      </div>
    )
  } catch (error) {
    console.error("Error parsing results:", error)
    return (
      <div className="overflow-x-auto bg-blue-50 p-3 rounded-md border border-blue-100">
        <p className="text-red-600 mb-2">Error parsing results</p>
        <pre className="text-xs text-blue-800">{JSON.stringify(results, null, 2)}</pre>
      </div>
    )
  }
}
