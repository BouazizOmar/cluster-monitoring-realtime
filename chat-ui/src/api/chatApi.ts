import axios from 'axios';

// Create an axios instance with default config
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:5001',
  headers: {
    'Content-Type': 'application/json',
  },
});



interface QueryResponse {
  success: boolean;
  sql_query?: string;
  results?: any;
  explanation?: string;
  error?: string;
}

export const processQuery = async (query: string): Promise<QueryResponse> => {
  try {
    const response = await apiClient.post<QueryResponse>('/api/process-query', { query });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response) {
      return error.response.data as QueryResponse;
    }
    
    return {
      success: false,
      error: 'Failed to connect to the server',
    };
  }
};