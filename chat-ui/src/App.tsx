import { useState, useRef, useEffect } from 'react'
import ChatMessage from './components/ChatMessage'
import ChatInput from './components/ChatInput'
import { processQuery } from './api/chatApi'

// Define types for our messages and responses
type MessageType = 'user' | 'bot'

interface Message {
  id: string
  type: MessageType
  text: string
  timestamp: Date
  results?: any
  query?: string
  explanation?: string
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'bot',
      text: 'Hello! I am your Snowflake Analytics Assistant. Ask me anything about your data!',
      timestamp: new Date()
    }
  ])
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Scroll to the bottom of the chat when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Function to handle sending a new message
  const handleSendMessage = async (text: string) => {
    if (!text.trim()) return

    // Add user message to the chat
    const userMessageId = Date.now().toString()
    const userMessage: Message = {
      id: userMessageId,
      type: 'user',
      text,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      // API call to your backend
      const response = await processQuery(text)
      
      if (response.success) {
        // Create bot response message
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          text: response.explanation || 'I processed your query.',
          timestamp: new Date(),
          results: response.results,
          query: response.sql_query,
          explanation: response.explanation
        }

        setMessages(prev => [...prev, botMessage])
      } else {
        // Add error message
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          text: response.error || 'Sorry, I encountered an error processing your query. Please try again.',
          timestamp: new Date()
        }
        
        setMessages(prev => [...prev, errorMessage])
      }
    } catch (error) {
      console.error('Error processing query:', error)
      
      // Add error message
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        text: 'Sorry, I encountered an error processing your query. Please try again.',
        timestamp: new Date()
      }
      
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="chat-container">
      <header className="header">
        <div className="logo">Snowflake Analytics Assistant</div>
      </header>

      <div className="chat-messages">
        {messages.map(message => (
          <ChatMessage 
            key={message.id} 
            message={message} 
          />
        ))}
        
        {isLoading && (
          <div className="message bot-message loading">
            <span>Thinking</span>
            <div className="loading-dots">
              <div className="dot"></div>
              <div className="dot"></div>
              <div className="dot"></div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
    </div>
  )
}

export default App