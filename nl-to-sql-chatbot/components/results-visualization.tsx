import React from "react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { BarChartComponent, PieChartComponent } from "@/components/ui/chat"

function formatSql(sql: string): string {
  // Simple SQL formatting - replace with more sophisticated formatting if needed
  return sql
    .replace(/\s+/g, " ")
    .replace(/\s*,\s*/g, ", ")
    .replace(/\s*;\s*/g, ";\n")
    .replace(/\s*(\()\s*/g, " $1")
    .replace(/\s*(\))\s*/g, "$1 ")
    .replace(/\bSELECT\b/gi, "\nSELECT")
    .replace(/\bFROM\b/gi, "\nFROM")
    .replace(/\bWHERE\b/gi, "\nWHERE")
    .replace(/\bAND\b/gi, "\n  AND")
    .replace(/\bOR\b/gi, "\n  OR")
    .replace(/\bGROUP BY\b/gi, "\nGROUP BY")
    .replace(/\bORDER BY\b/gi, "\nORDER BY")
    .replace(/\bHAVING\b/gi, "\nHAVING")
    .replace(/\bLIMIT\b/gi, "\nLIMIT")
    .replace(/\bJOIN\b/gi, "\nJOIN")
    .replace(/\bLEFT JOIN\b/gi, "\nLEFT JOIN")
    .replace(/\bRIGHT JOIN\b/gi, "\nRIGHT JOIN")
    .replace(/\bINNER JOIN\b/gi, "\nINNER JOIN")
    .replace(/\bOUTER JOIN\b/gi, "\nOUTER JOIN")
    .trim()
}

interface ResultsVisualizationProps {
  results: any
  message?: any // Optional message prop
}

export function ResultsVisualization({ results, message }: ResultsVisualizationProps) {
  if (!results) {
    return <p className="text-blue-800 p-4">No results available</p>
  }

  try {
    // Parse results if it's a string
    const parsedResults = typeof results === "string" ? JSON.parse(results) : results

    // Check if we have the expected structure
    if (!parsedResults.rows || !Array.isArray(parsedResults.rows)) {
      return (
        <div className="overflow-x-auto bg-blue-50 p-3 m-4 rounded-md border border-blue-100">
          <pre className="text-xs text-blue-800">{JSON.stringify(results, null, 2)}</pre>
        </div>
      )
    }

    // Prepare data for charts
    const osCounts = parsedResults.rows.reduce((acc: Record<string, number>, row: any) => {
      const osType = row.OS_TYPE || "Unknown"
      acc[osType] = (acc[osType] || 0) + 1
      return acc
    }, {})

    const osChartData = Object.entries(osCounts).map(([name, count]) => ({
      name,
      value: count,
    }))

    const vmChartData = parsedResults.rows.map((row: any) => ({
      name: row.VM_NAME || "Unknown",
      value: row.VM_KEY || 0,
    }))

    return (
      <div className="p-4 pt-3">
        <Tabs defaultValue="table">
          <TabsList className="w-full bg-gradient-to-r from-blue-100 to-blue-50 mb-3 p-0.5 h-10">
            <TabsTrigger
              value="table"
              className="flex items-center gap-1.5 data-[state=active]:bg-white rounded-sm flex-1 h-9"
            >
              Table
            </TabsTrigger>
            <TabsTrigger
              value="charts"
              className="flex items-center gap-1.5 data-[state=active]:bg-white rounded-sm flex-1 h-9"
            >
              Charts
            </TabsTrigger>
            <TabsTrigger
              value="cards"
              className="flex items-center gap-1.5 data-[state=active]:bg-white rounded-sm flex-1 h-9"
            >
              Cards
            </TabsTrigger>
          </TabsList>

          <TabsContent value="table" className="p-0 mt-0">
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
            <div className="mt-2 text-sm text-blue-600 flex items-center justify-between">
              <span>Total rows: {parsedResults.row_count || parsedResults.rows.length}</span>
            </div>
          </TabsContent>

          <TabsContent value="charts" className="p-0 mt-0">
            <div className="grid gap-4 md:grid-cols-2">
              <Card className="border-blue-100 shadow-sm">
                <CardHeader className="pb-2 bg-gradient-to-r from-blue-50 to-white">
                  <CardTitle className="text-blue-700 text-base">VMs by OS Type</CardTitle>
                  <CardDescription>Distribution of virtual machines by operating system</CardDescription>
                </CardHeader>
                <CardContent className="pt-4">
                  <PieChartComponent data={osChartData} />
                </CardContent>
              </Card>

              <Card className="border-blue-100 shadow-sm">
                <CardHeader className="pb-2 bg-gradient-to-r from-blue-50 to-white">
                  <CardTitle className="text-blue-700 text-base">VM Keys</CardTitle>
                  <CardDescription>VM keys for each virtual machine</CardDescription>
                </CardHeader>
                <CardContent className="pt-4">
                  <BarChartComponent data={vmChartData} />
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="cards" className="p-0 mt-0">
            <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
              {parsedResults.rows.map((row: any, index: number) => (
                <Card key={index} className="border-blue-100 shadow-sm hover:shadow-md transition-shadow">
                  <CardHeader className="bg-gradient-to-r from-blue-50 to-white pb-2">
                    <CardTitle className="text-blue-700 text-base">{row.VM_NAME || "Unknown VM"}</CardTitle>
                    <CardDescription>VM ID: {row.VM_KEY}</CardDescription>
                  </CardHeader>
                  <CardContent className="pt-3">
                    <dl className="grid grid-cols-2 gap-1 text-sm">
                      {Object.entries(row).map(([key, value]) => (
                        <React.Fragment key={key}>
                          <dt className="font-medium text-blue-600">{key}:</dt>
                          <dd className="text-blue-800">{String(value)}</dd>
                        </React.Fragment>
                      ))}
                    </dl>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    )
  } catch (error) {
    console.error("Error parsing results:", error)
    return (
      <div className="overflow-x-auto bg-blue-50 p-3 m-4 rounded-md border border-blue-100">
        <p className="text-red-600 mb-2">Error parsing results</p>
        <pre className="text-xs text-blue-800">{JSON.stringify(results, null, 2)}</pre>
      </div>
    )
  }
}
