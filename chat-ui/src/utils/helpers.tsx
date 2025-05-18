// Helper functions for the chat application

/**
 * Format a Date object to display time in 12-hour format (e.g., "10:04 AM")
 * @param date - Date object to format
 * @returns Formatted time string
 */
export const formatTime = (date: Date): string => {
    return date.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit'
    });
  };
  
  /**
   * Generate a unique ID for chat messages
   * @returns Unique string ID
   */
  export const generateId = (): string => {
    return Date.now().toString() + Math.random().toString(36).substr(2, 5);
  };
  
  /**
   * Format SQL query with proper indentation and syntax highlighting
   * This is a simple implementation - could be enhanced with a full SQL formatter library
   * @param sql - SQL query string
   * @returns Formatted SQL query
   */
  export const formatSqlQuery = (sql: string): string => {
    // This is a simple implementation - in production you might use a more sophisticated SQL formatter
    if (!sql) return '';
    
    // Basic cleaning
    return sql.trim();
  };
  
  /**
   * Check if object is empty
   * @param obj - Object to check
   * @returns Boolean indicating if object is empty
   */
  export const isEmptyObject = (obj: any): boolean => {
    return obj && Object.keys(obj).length === 0 && obj.constructor === Object;
  };
  
  /**
   * Create a debounced version of a function
   * @param func - Function to debounce
   * @param wait - Wait time in milliseconds
   * @returns Debounced function
   */
  export const debounce = (func: Function, wait: number): Function => {
    let timeout: ReturnType<typeof setTimeout>;
    
    return function executedFunction(...args: any[]) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  };