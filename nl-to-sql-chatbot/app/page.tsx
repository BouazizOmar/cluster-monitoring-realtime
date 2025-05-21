"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Send, Database, Snowflake } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { processQuery } from "@/lib/chat-api"
import { TableView } from "@/components/table-view"

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

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  sqlQuery?: string
  explanation?: string
  results?: any
  isProcessing?: boolean
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    // Add processing message
    const processingMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: "assistant",
      content: "Processing your query...",
      isProcessing: true,
    }
    setMessages((prev) => [...prev, processingMessage])

    try {
      const response = await processQuery(input.trim())

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.success
          ? "Here is the SQL query for your request:"
          : `Error: ${response.error || "Something went wrong"}`,
        sqlQuery: response.sql_query,
        explanation: response.explanation,
        results: response.results,
      }

      // Replace the processing message with the actual response
      setMessages((prev) => prev.map((msg) => (msg.isProcessing ? assistantMessage : msg)))
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Sorry, there was an error processing your request.",
      }

      // Replace the processing message with the error message
      setMessages((prev) => prev.map((msg) => (msg.isProcessing ? errorMessage : msg)))
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-b from-blue-50 to-white p-3">
      <Card className="w-full max-w-6xl shadow-md border-blue-100 bg-white overflow-hidden">
        <CardHeader className="border-b border-blue-100 bg-gradient-to-r from-blue-600 to-blue-500 text-white py-4">
          <CardTitle className="flex items-center gap-2 text-xl">
            <Snowflake className="h-6 w-6" />
            Snowflake SQL Assistant
          </CardTitle>
        </CardHeader>

        <CardContent className="p-0">
          <div className="h-[75vh] overflow-y-auto bg-white">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center p-6">
                <div className="bg-blue-50 p-8 rounded-full mb-6">
                  <Database className="h-16 w-16 text-blue-500" />
                </div>
                <h3 className="text-xl font-medium text-blue-700 mb-2">Ask me to query your Snowflake database</h3>
                <p className="text-blue-500 max-w-md">
                  Try questions like "Show me all Linux VMs" or "List the top 5 customers by revenue"
                </p>
                <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl">
                  {[
                    "Show me all Linux VMs",
                    "The average memory usage for each VM",
                    "The average CPU usage by mode for Ubuntu",
                    "Which service had the most failures on VM"
                  ].map((example) => (
                    <Button
                      key={example}
                      variant="outline"
                      className="text-blue-600 border-blue-200 hover:bg-blue-50 justify-start px-4 py-6 h-auto"
                      onClick={() => setInput(example)}
                    >
                      <span className="truncate">{example}</span>
                    </Button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="p-4 space-y-6">
                {messages.map((message) => (
                  <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div className={`max-w-[85%] ${message.role === "user" ? "order-1" : "order-none"}`}>
                      <div
                        className={`px-4 py-3 rounded-2xl shadow-sm ${
                          message.role === "user"
                            ? "bg-blue-600 text-white rounded-tr-none"
                            : message.isProcessing
                              ? "bg-blue-50 text-blue-800 border border-blue-100 rounded-tl-none"
                              : "bg-blue-50 text-blue-800 border border-blue-100 rounded-tl-none"
                        }`}
                      >
                        <p className="leading-relaxed">{message.content}</p>
                        {message.isProcessing && (
                          <div className="flex items-center gap-1.5 mt-2">
                            <div
                              className="h-1.5 w-1.5 bg-blue-400 rounded-full animate-bounce"
                              style={{ animationDelay: "0ms" }}
                            ></div>
                            <div
                              className="h-1.5 w-1.5 bg-blue-400 rounded-full animate-bounce"
                              style={{ animationDelay: "300ms" }}
                            ></div>
                            <div
                              className="h-1.5 w-1.5 bg-blue-400 rounded-full animate-bounce"
                              style={{ animationDelay: "600ms" }}
                            ></div>
                          </div>
                        )}
                      </div>

                      {message.role === "assistant" && message.sqlQuery && (
                        <div className="mt-3 space-y-4">
                          {/* SQL Query Section */}
                          <div className="rounded-md border border-blue-100 overflow-hidden shadow-sm">
                            <div className="bg-blue-50 px-4 py-2 border-b border-blue-100">
                              <h3 className="font-medium text-blue-700">SQL Query</h3>
                            </div>
                            <div className="p-4">
                              <div className="bg-blue-50 p-3 rounded-md overflow-x-auto text-blue-800 border border-blue-100">
                                <pre className="whitespace-pre-wrap break-words text-sm" style={{ maxWidth: "100%" }}>
                                  <code>{formatSql(message.sqlQuery || "")}</code>
                                </pre>
                              </div>
                            </div>
                          </div>

                          {/* Explanation Section */}
                          {message.explanation && (
                            <div className="rounded-md border border-blue-100 overflow-hidden shadow-sm">
                              <div className="bg-blue-50 px-4 py-2 border-b border-blue-100">
                                <h3 className="font-medium text-blue-700">Explanation</h3>
                              </div>
                              <div className="p-4">
                                <div className="bg-blue-50 p-3 rounded-md border border-blue-100 text-blue-800">
                                  {message.explanation}
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Results Section */}
                          {message.results && (
                            <div className="rounded-md border border-blue-100 overflow-hidden shadow-sm">
                              <div className="bg-blue-50 px-4 py-2 border-b border-blue-100">
                                <h3 className="font-medium text-blue-700">Results</h3>
                              </div>
                              <div className="p-4">
                                <TableView results={message.results} />
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </CardContent>

        <CardFooter className="border-t border-blue-100 p-4 bg-gradient-to-r from-blue-50 to-white">
          <form onSubmit={handleSubmit} className="flex w-full gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about your data..."
              disabled={isLoading}
              className="flex-grow border-blue-200 focus-visible:ring-blue-400 py-6 px-4 text-base shadow-sm"
            />
            <Button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="bg-blue-600 hover:bg-blue-700 text-white px-5"
            >
              {isLoading ? (
                <div className="h-5 w-5 animate-spin rounded-full border-2 border-white border-t-transparent" />
              ) : (
                <Send className="h-5 w-5" />
              )}
            </Button>
          </form>
        </CardFooter>
      </Card>
    </div>
  )
}
