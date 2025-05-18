const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
const PORT = 5000;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Sample response data
const sampleResults = {
  'service_name': ['nginx', 'mongodb', 'redis', 'mysql', 'postgresql'],
  'usage_count': [145, 120, 98, 76, 65],
  'avg_cpu_usage': [32.5, 28.7, 22.1, 18.4, 15.9]
};

// Mock API endpoint
app.post('/api/process-query', (req, res) => {
  const { query } = req.body;
  
  // Simulate processing time
  setTimeout(() => {
    res.json({
      success: true,
      sql_query: `SELECT 
  service_name, 
  COUNT(*) as usage_count,
  AVG(cpu_usage) as avg_cpu_usage
FROM 
  service_metrics
WHERE 
  timestamp >= CURRENT_DATE() - INTERVAL '7 days'
GROUP BY 
  service_name
ORDER BY 
  usage_count DESC
LIMIT 5;`,
      results: sampleResults,
      explanation: `Based on your query "${query}", I analyzed the service usage data from the past 7 days. NGINX has been the most frequently used service with 145 occurrences, followed by MongoDB (120) and Redis (98). The chart shows both the usage count and average CPU usage for each service.`
    });
  }, 1500);
});

// Start the server
app.listen(PORT, () => {
  console.log(`Mock server running on port ${PORT}`);
}); 