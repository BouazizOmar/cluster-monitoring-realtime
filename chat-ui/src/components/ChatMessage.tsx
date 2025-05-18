import ResultsDisplay from './ResultsDisplay';

interface Message {
  id: string;
  type: 'user' | 'bot';
  text: string;
  timestamp: Date;
  results?: any;
  query?: string;
  explanation?: string;
}

interface ChatMessageProps {
  message: Message;
}

const ChatMessage = ({ message }: ChatMessageProps) => {
  const { type, text, results, query } = message;
  const isBot = type === 'bot';

  return (
    <div className={`message ${isBot ? 'bot-message' : 'user-message'}`}>
      <div className="message-text">{text}</div>
      
      {isBot && results && (
        <div className="message-results">
          {query && (
            <div className="sql-query">
              <strong>Generated SQL:</strong>
              <pre className="sql-query-content">{query}</pre>
            </div>
          )}
          <ResultsDisplay results={results} />
        </div>
      )}
    </div>
  );
};

export default ChatMessage;